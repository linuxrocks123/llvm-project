#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# This script performs some setup and contains some utilities used for in the
# monolithic-linux.sh and monolithic-windows.sh scripts.

set -ex
set -o pipefail

MONOREPO_ROOT="${MONOREPO_ROOT:="$(git rev-parse --show-toplevel)"}"
BUILD_DIR="${BUILD_DIR:=${MONOREPO_ROOT}/build}"

rm -rf "${BUILD_DIR}"

sccache --zero-stats

function at-exit {
  retcode=$?

  mkdir -p artifacts
  sccache --show-stats
  sccache --show-stats >> artifacts/sccache_stats.txt
  cp "${MONOREPO_ROOT}"/*.ninja_log artifacts/ || :
  cp "${MONOREPO_ROOT}"/*.log artifacts/ || :
  cp "${BUILD_DIR}"/test-results.*.xml artifacts/ || :

  # If building fails there will be no results files.
  shopt -s nullglob

  if [[ -n "$GITHUB_ACTIONS" ]]; then
    python "${MONOREPO_ROOT}"/.ci/generate_test_report_github.py \
      $retcode "${BUILD_DIR}"/test-results.*.xml "${MONOREPO_ROOT}"/ninja*.log \
      >> $GITHUB_STEP_SUMMARY
    # The premerge advisor runs inside LLVM's internal cluster and is
    # unreachable from standard GitHub runners, so skip it and exit with the
    # real build result.
    exit $retcode
  fi
}
trap at-exit EXIT

function start-group {
  groupname=$1
  if [[ -n "$GITHUB_ACTIONS" ]]; then
    echo "::endgroup"
    echo "::group::$groupname"
  elif [[ -n "$POSTCOMMIT_CI" ]]; then
    echo "@@@$STEP@@@"
  else
    echo "Starting $groupname"
  fi
}

export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r "${MONOREPO_ROOT}"/.ci/all_requirements.txt
