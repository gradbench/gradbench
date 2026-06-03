EXECUTABLES=hello ba det gmm kmeans llsq lse lstm ode particle

# include/ad.hpp must be present before `make` runs (the nix tool wrapper
# materializes it from a pinned fetchurl); we don't ship a rule to download
# it here because cpp.py invokes `make -B`, which forces every rule to fire
# on every compile and would re-`wget` -- failing without network/`wget`,
# even though the header is already on disk.
EXTRA_DEPS=include/ad.hpp

include ../../cpp/common.mk

CXXFLAGS+=-Iinclude

bin/kmeans:
	echo "kmeans temporarily disabled, see #654"
	exit 1

