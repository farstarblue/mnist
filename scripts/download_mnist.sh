#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

choose_fetcher() {
    if command -v curl >/dev/null 2>&1; then
        echo curl
    elif command -v wget >/dev/null 2>&1; then
        echo wget
    else
        echo "需要 curl 或 wget 之一来下载 MNIST 数据集。" >&2
        exit 1
    fi
}

FETCHER="$(choose_fetcher)"

fetch_file() {
    local url="$1"
    local target="$2"

    if [[ "${FETCHER}" == "curl" ]]; then
        curl -fsSL --retry 3 --connect-timeout 15 "${url}" -o "${target}"
    else
        wget -q --tries=3 --timeout=15 -O "${target}" "${url}"
    fi
}

check_size() {
    local file="$1"
    local expected="$2"
    local actual
    actual=$(stat -c '%s' "${file}")
    [[ "${actual}" == "${expected}" ]]
}

ensure_file() {
    local name="$1"
    local expected_size="$2"
    local tmp_gz="${DATA_DIR}/${name}.gz"
    local output="${DATA_DIR}/${name}"
    local base_urls=(
        "${MNIST_BASE_URL:-}"
        "https://mirror.ghproxy.com/https://raw.githubusercontent.com/fgnt/mnist/master"
        "https://ghproxy.cn/https://raw.githubusercontent.com/fgnt/mnist/master"
        "https://raw.githubusercontent.com/fgnt/mnist/master"
        "https://ossci-datasets.s3.amazonaws.com/mnist"
        "https://storage.googleapis.com/cvdf-datasets/mnist"
    )

    if [[ -f "${output}" ]] && check_size "${output}" "${expected_size}"; then
        echo "已存在，跳过: ${name}"
        return 0
    fi

    rm -f "${tmp_gz}" "${output}"

    for base_url in "${base_urls[@]}"; do
        [[ -z "${base_url}" ]] && continue
        echo "尝试下载: ${base_url}/${name}.gz"
        if fetch_file "${base_url}/${name}.gz" "${tmp_gz}"; then
            if gzip -dc "${tmp_gz}" > "${output}" && check_size "${output}" "${expected_size}"; then
                rm -f "${tmp_gz}"
                echo "下载完成: ${name}"
                return 0
            fi
        fi
        rm -f "${tmp_gz}" "${output}"
    done

    echo "下载失败: ${name}" >&2
    exit 1
}

ensure_file train-images-idx3-ubyte 47040016
ensure_file train-labels-idx1-ubyte 60008
ensure_file t10k-images-idx3-ubyte 7840016
ensure_file t10k-labels-idx1-ubyte 10008

echo "MNIST 数据集已就绪: ${DATA_DIR}"
