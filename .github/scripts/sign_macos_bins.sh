#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

WHEEL_FILE="$1"
DEST_DIR="$2"
IDENTITY_ID="Developer ID Application: The Khronos Group, Inc. (TD2656HYNK)"

if [[ -z "$WHEEL_FILE" || -z "$DEST_DIR" ]]; then
  echo "Usage: $0 <wheel_file> <dest_dir>"
  exit 1
fi

if [[ -z "$BUILD_CERTIFICATE_BASE64" ]]; then
  echo "Error: BUILD_CERTIFICATE_BASE64 environment variable is not set."
  exit 1
fi

if [[ -z "$P12_PASSWORD" ]]; then
  echo "Error: P12_PASSWORD environment variable is not set."
  exit 1
fi

if [[ -z "$KEYCHAIN_PASSWORD" ]]; then
  echo "Error: KEYCHAIN_PASSWORD environment variable is not set."
  exit 1
fi

# Ensure the destination directory exists
mkdir -p "${DEST_DIR}"

# Name of the wheel (e.g., slangpy-0.27.0-cp311-cp311-macosx_14_0_arm64.whl)
WHEEL_BASENAME=$(basename "${WHEEL_FILE}")

# Temporary directory for unzipping and processing
RUNNER_TEMP=$(mktemp -d ./wheel_repair_XXXXXXXXXX)
echo "Temporary directory: ${RUNNER_TEMP}"

# Cleanup function to remove the temporary directory on exit
cleanup() {
  echo "Cleaning up temporary directory: ${RUNNER_TEMP}"
  rm -rf "${RUNNER_TEMP}"
}
trap cleanup EXIT

# Unzip the wheel into the temporary directory
echo "Unzipping wheel: ${WHEEL_FILE} to ${RUNNER_TEMP}/${WHEEL_BASENAME%.whl}"
unzip -q "${WHEEL_FILE}" -d "${RUNNER_TEMP}/${WHEEL_BASENAME%.whl}"

# Find and sign dylibs and other executables
# Add other patterns here if you have other executables, e.g., -o -name 'my_executable'
echo "Searching for files to sign in ${RUNNER_TEMP}/${WHEEL_BASENAME%.whl}"
# Initialize an empty array to store binary files
declare -a binaries

# Find the binaries in the wheel to sign
echo "Finding binaries..."
while IFS= read -r -d $'\0' file_to_sign; do
  if [[ -f "$file_to_sign" ]]; then
    binaries+=("$file_to_sign")
    echo "Found binary: $file_to_sign"
  else
    echo "Skipping (not a file?): $file_to_sign"
  fi
done < <(find "${RUNNER_TEMP}/${WHEEL_BASENAME%.whl}" \( -name '*.dylib' -o -name '*.so' \) -print0)

echo "Found ${#binaries[@]} binaries to sign."
echo "${binaries[@]}"

# Import the signing identity
import_certificate() {
    CERTIFICATE_PATH=$RUNNER_TEMP/build_certificate.p12
    KEYCHAIN_PATH=$RUNNER_TEMP/app-signing.keychain-db

    # import certificate and provisioning profile from secrets
    echo -n "$BUILD_CERTIFICATE_BASE64" | base64 --decode --output "$CERTIFICATE_PATH"
    # create temporary keychain
    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
    security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH"
    # import certificate to keychain
    security import "$CERTIFICATE_PATH" -P "$P12_PASSWORD" -A -t cert -f pkcs12 -k "$KEYCHAIN_PATH"
    security list-keychain -d user -s "$KEYCHAIN_PATH"
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "${KEYCHAIN_PASSWORD}" "$KEYCHAIN_PATH"
}

sign_bin() {
    binaries=$1
    # Sign main binaries
     for b in "${binaries[@]}"; do
       if [[ -f "$b" ]]; then
         echo "Signing binary '$b'..."
         /usr/bin/codesign --force --options runtime -s "${IDENTITY_ID}" "$b" -v
       fi
     done
     echo "Signing process complete."
}

import_certificate;
sign_bin "${binaries[@]}";

# Re-zip the wheel from the modified contents
echo "Re-zipping wheel to ${DEST_DIR}/${WHEEL_BASENAME}"
(cd "${RUNNER_TEMP}/${WHEEL_BASENAME%.whl}" && zip -r -q "${DEST_DIR}/${WHEEL_BASENAME}" ./*)

echo "Wheel repair complete. Signed wheel at: ${DEST_DIR}/${WHEEL_BASENAME}"
