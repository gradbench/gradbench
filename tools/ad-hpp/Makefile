EXECUTABLES=hello ba det gmm kmeans llsq lse lstm ode particle

EXTRA_DEPS=include/ad.hpp

include ../../cpp/common.mk

GIT_TAG=v1.7.1
CXXFLAGS+=-Iinclude

include/ad.hpp:
	mkdir -p include
	wget https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/$(GIT_TAG)/$@ -O $@

include/extensions/ad_interoperability_eigen.hpp:
	mkdir -p include/extensions
	wget https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/$(GIT_TAG)/$@ -O $@

