FROM mcr.microsoft.com/dotnet/sdk:8.0
WORKDIR /home/gradbench

# https://diffsharp.github.io/install.html
RUN dotnet new console -lang "F#"
RUN dotnet add package DiffSharp-lite
RUN dotnet add package Newtonsoft.Json

COPY tools/diffsharp/ .

# run f# through dotnet and f# interactive
ENTRYPOINT ["dotnet", "fsi", "run.fsx"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
