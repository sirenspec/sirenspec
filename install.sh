#!/usr/bin/env sh
set -e

PACKAGE="sirenspec"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

check_python() {
    if ! command -v python3 > /dev/null 2>&1; then
        echo "error: Python 3.${MIN_PYTHON_MINOR}+ is required but python3 was not found." >&2
        echo "Install Python from https://python.org and try again." >&2
        exit 1
    fi

    version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    major=$(python3 -c "import sys; print(sys.version_info.major)")
    minor=$(python3 -c "import sys; print(sys.version_info.minor)")

    if [ "$major" -lt "$MIN_PYTHON_MAJOR" ] || { [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -lt "$MIN_PYTHON_MINOR" ]; }; then
        echo "error: Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required (found ${version})." >&2
        exit 1
    fi
}

install_with_uv() {
    echo "Installing ${PACKAGE} with uv..."
    uv tool install "$PACKAGE"
}

install_with_pipx() {
    echo "Installing ${PACKAGE} with pipx..."
    pipx install "$PACKAGE"
}

install_with_pip() {
    echo "Installing ${PACKAGE} with pip..."
    python3 -m pip install --user "$PACKAGE"
}

verify_install() {
    if command -v sirenspec > /dev/null 2>&1; then
        sirenspec --version
    else
        echo "warning: sirenspec was installed but is not on PATH." >&2
        echo "You may need to add the install location to your PATH." >&2
    fi
}

check_python

if command -v uv > /dev/null 2>&1; then
    install_with_uv
elif command -v pipx > /dev/null 2>&1; then
    install_with_pipx
elif command -v pip3 > /dev/null 2>&1 || python3 -m pip --version > /dev/null 2>&1; then
    install_with_pip
else
    echo "error: no suitable installer found (uv, pipx, or pip required)." >&2
    echo "Install uv (https://docs.astral.sh/uv/) or pip and try again." >&2
    exit 1
fi

verify_install

echo ""
echo "SirenSpec installed successfully!"
echo "Run 'sirenspec --help' to get started."
