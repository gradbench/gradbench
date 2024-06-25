FROM mcr.microsoft.com/dotnet/sdk:6.0 
WORKDIR /home/gradbench

# https://diffsharp.github.io/install.html
RUN dotnet new console -lang "F#"
RUN dotnet add package DiffSharp-cpu
RUN dotnet add package FSharp.Data 

COPY functions.fsx .
COPY run.fsx .

# run f# through dotnet and f# interactive
ENTRYPOINT ["dotnet", "fsi", "run.fsx"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

