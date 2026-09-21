// SPDX-FileCopyrightText: 2024-2026 Temps Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Archival backend for finished build/deploy job logs.
//!
//! Build/deploy logs are written incrementally to a local JSONL scratch file
//! while a job is running (see [`crate::structured_logs::StructuredLogService`]),
//! because S3 has no true append operation and live tailing during an
//! in-progress job needs to stay fast. Once a job reaches a terminal state,
//! [`crate::file_logs::LogService::archive_log`] uploads the finished file to
//! this backend as a single object and deletes the local copy, so local disk
//! only ever holds logs for currently-running jobs rather than a whole
//! deployment history.
//!
//! This mirrors the storage-backend pattern in `temps-log-aggregator`
//! (`LogStorage` trait + `S3Storage`), but the shape of the problem is
//! different: the aggregator writes many small compressed chunks per
//! project/service/hour and needs range reads, while a build/deploy log is a
//! single whole JSONL file per job read back in one piece. Hence a narrower
//! trait (`upload_log` / `download_log`) rather than reusing `LogStorage`
//! directly.

use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::Config;
use tracing::debug;

/// Errors from the log archive storage backend.
#[derive(Debug, thiserror::Error)]
pub enum LogArchiveStorageError {
    #[error("Failed to upload archived log '{key}' to bucket '{bucket}': {reason}")]
    Upload {
        bucket: String,
        key: String,
        reason: String,
        /// Whether retrying this exact upload might succeed. `false` for
        /// permanent failures (bad credentials, a bucket that doesn't exist,
        /// malformed configuration) that will fail identically on every
        /// attempt -- `LogService::archive_log`'s retry loop checks this and
        /// stops immediately rather than spending its remaining attempts
        /// (and the delay between them) on an error that cannot resolve
        /// itself. See CLAUDE.md's "Resilience Patterns" section: retrying
        /// authentication/not-found failures is explicitly called out as
        /// something to avoid.
        retryable: bool,
    },

    #[error("Failed to download archived log '{key}' from bucket '{bucket}': {reason}")]
    Download {
        bucket: String,
        key: String,
        reason: String,
    },

    #[error("Archived log '{key}' not found in bucket '{bucket}'")]
    NotFound { bucket: String, key: String },
}

impl LogArchiveStorageError {
    /// Whether retrying the operation that produced this error stands a
    /// chance of succeeding. `Upload` carries its own classification (set at
    /// the point the underlying S3 SDK error is known); every other variant
    /// is either not part of the upload retry path (`Download`) or
    /// definitionally permanent (`NotFound`).
    pub fn is_retryable(&self) -> bool {
        match self {
            LogArchiveStorageError::Upload { retryable, .. } => *retryable,
            LogArchiveStorageError::Download { .. } => false,
            LogArchiveStorageError::NotFound { .. } => false,
        }
    }
}

/// S3 error codes that indicate a permanent failure -- retrying with the
/// same credentials/bucket/key will fail identically every time. Anything
/// not in this list (5xx service errors, throttling, or no error code at
/// all because the request never reached S3 -- a timeout or connection
/// failure) is treated as potentially transient and left retryable.
const PERMANENT_S3_ERROR_CODES: &[&str] = &[
    "AccessDenied",
    "AllAccessDisabled",
    "AuthorizationHeaderMalformed",
    "ExpiredToken",
    "InvalidAccessKeyId",
    "InvalidBucketName",
    "InvalidToken",
    "NoSuchBucket",
    "SignatureDoesNotMatch",
];

/// Classify an S3 SDK error as retryable or permanent from its error code.
/// `err.code()` (via [`ProvideErrorMetadata`]) is only populated for
/// responses S3 actually returned (`SdkError::ServiceError`); construction,
/// dispatch, timeout and malformed-response failures never reached S3 at
/// all and are conservatively treated as transient network conditions.
fn is_retryable_s3_error(err: &impl ProvideErrorMetadata) -> bool {
    match err.code() {
        Some(code) => !PERMANENT_S3_ERROR_CODES.contains(&code),
        None => true,
    }
}

/// Pluggable archive backend for finished build/deploy logs.
///
/// Implementations must be safe to share across threads and async tasks.
/// There is currently one production implementation, [`S3LogArchive`]; when
/// no backend is configured `LogService` holds `None` instead of a no-op
/// implementation, so archival (and the local-file deletion it triggers) can
/// never run for an install that never opted in.
#[async_trait::async_trait]
pub trait LogArchiveStorage: Send + Sync + 'static {
    /// Upload the full contents of a finished log as one object.
    async fn upload_log(&self, key: &str, data: Vec<u8>) -> Result<(), LogArchiveStorageError>;

    /// Download the full contents of a previously archived log.
    async fn download_log(&self, key: &str) -> Result<Vec<u8>, LogArchiveStorageError>;
}

/// S3-compatible archive backend for build/deploy logs.
///
/// Works with AWS S3, MinIO, Tigris, Cloudflare R2, RustFS, and any other
/// S3-compatible API -- the same set of backends `temps-log-aggregator`'s
/// `S3Storage` supports, since both are built from the same
/// `temps_core::LogStorageConfig::S3` variant.
pub struct S3LogArchive {
    client: S3Client,
    bucket: String,
    prefix: Option<String>,
}

impl S3LogArchive {
    /// Build a new S3 archive backend from raw connection fields (the fields
    /// of `temps_core::LogStorageConfig::S3`). Construction never fails: it
    /// only builds an SDK client config, it does not perform I/O or validate
    /// credentials against the bucket.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bucket: String,
        prefix: Option<String>,
        region: String,
        endpoint: Option<String>,
        access_key_id: String,
        secret_access_key: String,
        force_path_style: bool,
    ) -> Self {
        let creds = Credentials::new(access_key_id, secret_access_key, None, None, "temps-logs");

        let mut s3_config = Config::builder()
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
            .region(Region::new(region.clone()))
            .credentials_provider(creds)
            .force_path_style(force_path_style);

        if let Some(endpoint_url) = &endpoint {
            s3_config = s3_config.endpoint_url(endpoint_url);
        }

        let client = S3Client::from_conf(s3_config.build());

        debug!(bucket = %bucket, region = %region, "S3 build/deploy log archive initialized");

        Self {
            client,
            bucket,
            prefix,
        }
    }

    /// Build the full S3 key including the optional configured prefix.
    fn full_key(&self, key: &str) -> String {
        match &self.prefix {
            Some(prefix) => format!("{}/{}", prefix.trim_end_matches('/'), key),
            None => key.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl LogArchiveStorage for S3LogArchive {
    async fn upload_log(&self, key: &str, data: Vec<u8>) -> Result<(), LogArchiveStorageError> {
        let full_key = self.full_key(key);
        let body = ByteStream::from(data);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .body(body)
            .content_type("application/jsonl")
            .send()
            .await
            .map_err(|e| {
                let retryable = is_retryable_s3_error(&e);
                LogArchiveStorageError::Upload {
                    bucket: self.bucket.clone(),
                    key: full_key.clone(),
                    reason: e.to_string(),
                    retryable,
                }
            })?;

        debug!(bucket = %self.bucket, key = %full_key, "Archived build/deploy log to S3");
        Ok(())
    }

    async fn download_log(&self, key: &str) -> Result<Vec<u8>, LogArchiveStorageError> {
        let full_key = self.full_key(key);

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&full_key)
            .send()
            .await
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("NoSuchKey") || err_str.contains("404") {
                    LogArchiveStorageError::NotFound {
                        bucket: self.bucket.clone(),
                        key: full_key.clone(),
                    }
                } else {
                    LogArchiveStorageError::Download {
                        bucket: self.bucket.clone(),
                        key: full_key.clone(),
                        reason: err_str,
                    }
                }
            })?;

        let data = response
            .body
            .collect()
            .await
            .map_err(|e| LogArchiveStorageError::Download {
                bucket: self.bucket.clone(),
                key: full_key.clone(),
                reason: format!("failed reading response body: {e}"),
            })?
            .into_bytes()
            .to_vec();

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_archive(prefix: Option<&str>) -> S3LogArchive {
        S3LogArchive::new(
            "test-bucket".to_string(),
            prefix.map(str::to_string),
            "us-east-1".to_string(),
            None,
            "id".to_string(),
            "secret".to_string(),
            false,
        )
    }

    #[test]
    fn test_full_key_with_prefix() {
        let archive = test_archive(Some("logs/"));
        assert_eq!(
            archive.full_key("build-logs/deployment-1-job-build.log"),
            "logs/build-logs/deployment-1-job-build.log"
        );
    }

    #[test]
    fn test_full_key_without_prefix() {
        let archive = test_archive(None);
        assert_eq!(
            archive.full_key("build-logs/deployment-1-job-build.log"),
            "build-logs/deployment-1-job-build.log"
        );
    }

    #[test]
    fn test_full_key_trims_trailing_slash_in_prefix() {
        let archive = test_archive(Some("logs"));
        assert_eq!(
            archive.full_key("build-logs/x.log"),
            "logs/build-logs/x.log"
        );
    }
}
