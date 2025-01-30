# DiffSharp

[DiffSharp][] is a differentiable programming library for the [F#][] programming language.

[diffsharp]: https://diffsharp.github.io/
[f#]: https://fsharp.org/

## Running outside of Docker

To run this tool outside of Docker, install the .NET SDK (which
provides the `dotnet` command), the F# compiler, and run:

```
$ dotnet run -c release --project tools/diffsharp/diffsharp.fsproj
```
