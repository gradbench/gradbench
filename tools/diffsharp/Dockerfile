FROM mcr.microsoft.com/dotnet/sdk:8.0
WORKDIR /home/gradbench

COPY tools/diffsharp/ tools/diffsharp

RUN dotnet build tools/diffsharp

# run f# through dotnet and f# interactive
ENTRYPOINT ["dotnet", "run", "-c", "release", "--project", "tools/diffsharp/diffsharp.fsproj"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
