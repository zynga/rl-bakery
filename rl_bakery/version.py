"""Define feature_generator version information."""

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '6'
_PATCH_VERSION = '0'

# When building releases, we can update this value on the release branch to
# reflect the current release candidate ('rc0', 'rc1') or, finally, the official
# stable release (indicated by `_REL_SUFFIX = ''`). Outside the context of a
# release branch, the current version is by default assumed to be a
# 'development' version, labeled 'dev'.
_DEV_SUFFIX = 'dev'
_REL_SUFFIX = 'rc1'

# Example, '0.4.0.rc0'
__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
])
__dev_version__ = '{}.{}'.format(__version__, _DEV_SUFFIX)
__rel_version__ = '{}.{}'.format(__version__, _REL_SUFFIX)
