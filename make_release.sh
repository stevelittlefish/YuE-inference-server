#!/bin/bash
#
# Release script for the YuE2 inference server (ASS backend fork).
# Usage: ./make_release.sh <version> <message>
# Example: ./make_release.sh v1.0.0 "Initial release"
#
# Creates an annotated vX.Y.Z tag and pushes it. Pushing the tag fires the
# container workflow (.github/workflows/container.yml), which builds the Docker
# image and publishes it to
# ghcr.io/stevelittlefish/yue-inference-server. Same shape as the ACE-Step /
# stem-separator / stable-audio-3 release scripts this was copied from.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ $# -ne 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <version> <message>"
    echo ""
    LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || git tag -l "v*" | sort -V | tail -1)
    if [ -n "$LATEST_TAG" ]; then
        echo "Latest tag: $LATEST_TAG"
        echo ""
    fi
    echo "Example:"
    echo "  $0 v1.0.0 \"Initial release\""
    echo "  $0 v1.2.3 \"Bug fixes and improvements\""
    echo ""
    exit 1
fi

VERSION=$1
MESSAGE=$2

if [[ ! $VERSION =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-.*)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo ""
    echo "Version must follow the pattern: v<major>.<minor>.<patch>"
    echo "Examples: v1.0.0, v2.3.1, v1.0.0-beta, v2.0.0-rc1"
    echo ""
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    read -p "Do you want to continue anyway? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: You are on branch '$CURRENT_BRANCH', not 'main'${NC}"
    read -p "Do you want to continue anyway? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag $VERSION already exists${NC}"
    echo ""
    echo "Existing tags:"
    git tag -l "v*" | tail -5
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}=== Release Summary ===${NC}"
echo "Version:  $VERSION"
echo "Message:  $MESSAGE"
echo "Branch:   $CURRENT_BRANCH"
echo ""

read -p "Create and push this release? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo -e "${GREEN}Creating release...${NC}"

git tag -a "$VERSION" -m "$MESSAGE"
echo -e "${GREEN}✓${NC} Created tag $VERSION"

git push origin "$VERSION"
echo -e "${GREEN}✓${NC} Pushed tag to origin"

echo ""
echo -e "${GREEN}=== Release Created Successfully! ===${NC}"
echo ""
echo "GitHub Actions will now build and push the Docker image to GHCR."
echo "Check the progress at:"
echo "  https://github.com/stevelittlefish/YuE-inference-server/actions"
echo ""
echo "Once complete, pull it with:"
echo "  docker pull ghcr.io/stevelittlefish/yue-inference-server:$VERSION"
echo "  docker pull ghcr.io/stevelittlefish/yue-inference-server:latest"
echo ""
