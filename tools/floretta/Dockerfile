FROM rust
ARG TARGETARCH
RUN apt-get update && apt-get install -y musl-tools
WORKDIR /home/gradbench
COPY tools/floretta tools/floretta
RUN rustup target add "$(tools/floretta/target.py $TARGETARCH)"
COPY Cargo.toml .
COPY Cargo.lock .
COPY crates/gradbench-floretta crates/gradbench-floretta
RUN cargo build --release --target "$(tools/floretta/target.py $TARGETARCH)"
RUN cp "target/$(tools/floretta/target.py $TARGETARCH)/release/gradbench-floretta" .

FROM scratch
WORKDIR /home/gradbench
COPY tools/floretta tools/floretta
COPY --from=0 /home/gradbench/gradbench-floretta .
ENTRYPOINT ["./gradbench-floretta"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
