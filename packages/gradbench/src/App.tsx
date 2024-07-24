import "./App.css";

const App = () => {
  return (
    <>
      <a
        href="https://github.com/gradbench/gradbench"
        target="_blank"
        aria-label="View source on GitHub"
      >
        <svg
          className="github-corner"
          width="89"
          height="89"
          viewBox="0 0 250 250"
          style={{
            color: "#3e4756",
            position: "absolute",
            top: 0,
            border: 0,
            right: 0,
          }}
          aria-hidden="true"
        >
          <defs>
            <mask id="octo-mask" x="0" y="0" width="250" height="250">
              <rect x="0" y="0" width="250" height="250" fill="black" />
              <path
                d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"
                fill="white"
              />
              <path
                d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2"
                fill="black"
                style={{ transformOrigin: "130px 106px" }}
                className="octo-arm"
              />
              <path
                d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z"
                fill="black"
                className="octo-body"
              />
            </mask>
          </defs>
          <rect
            x="0"
            y="0"
            width="250"
            height="250"
            fill="currentColor"
            mask="url(#octo-mask)"
          />
        </svg>
      </a>
      <h1 className="header">
        GradBench
        <span className="subtitle">
          A Benchmark for Differentiable Programming Across Languages and
          Domains
        </span>
      </h1>

      {/* Appears for mobile view only */}
      <p className="hidden">
        <b style={{ color: "#3e4756" }}>
          A Benchmark for Differentiable Programming Across Languages and
          Domains
        </b>
      </p>

      <div className="row">
        <h2 className="subheading">Benchmarking</h2>
        <div className="fillspace"></div>
      </div>

      <ul>
        <li> GradBench is designed to work across languages and domains. </li>
        <li>
          Functions are written in <b style={{ color: "#3e4756" }}>Adroit</b>{" "}
          and automatically translated to a tool's native language.
        </li>
        <li>
          The architecture allows for the easy addition of new functions and
          tools.
        </li>
      </ul>

      <div className="row">
        <h2 className="subheading">Adroit</h2>
        <div className="fillspace"></div>
      </div>

      <p className="text">TODO: Explain Adroit</p>

      <div className="row">
        <h2 className="subheading">Currently Implemented</h2>
        <div className="fillspace"></div>
      </div>

      <div className="container">
        <div className="tbody">
          <table>
            <tr>
              <th className="no-border"></th>
              <th className="tool">
                <a href="https://github.com/HIPS/autograd" target="_blank">
                  Autograd
                </a>
              </th>
              <th className="tool">
                <a
                  href="https://diffsharp.github.io/index.html"
                  target="_blank"
                >
                  DiffSharp
                </a>
              </th>
              <th className="tool">
                <a
                  href="https://jax.readthedocs.io/en/latest/index.html"
                  target="_blank"
                >
                  JAX
                </a>
              </th>
              <th className="tool">
                <a
                  href="https://mygrad.readthedocs.io/en/latest/"
                  target="_blank"
                >
                  MyGrad
                </a>
              </th>
              <th className="tool">
                <a href="https://pytorch.org/" target="_blank">
                  PyTorch
                </a>
              </th>
              <th className="tool">
                <a href="https://github.com/lecopivo/SciLean" target="_blank">
                  SciLean
                </a>
              </th>
              <th className="tool">
                <a
                  href="https://tapenade.gitlabpages.inria.fr/userdoc/build/html/index.html"
                  target="_blank"
                >
                  Tapenade
                </a>
              </th>
              <th className="tool">
                <a href="https://www.tensorflow.org/" target="_blank">
                  TensorFlow
                </a>
              </th>
              <th className="tool">
                <a href="https://fluxml.ai/Zygote.jl/stable/" target="_blank">
                  Zygote
                </a>
              </th>
            </tr>
            <tr>
              <td className="module">
                {" "}
                <a
                  href="https://github.com/gradbench/gradbench/tree/main/evals/hello"
                  target="_blank"
                >
                  Hello
                </a>
              </td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
              <td className="emoji">&#10004;</td>
            </tr>
            <tr>
              <td className="module">
                <a
                  href="https://github.com/gradbench/gradbench/tree/main/evals/gmm"
                  target="_blank"
                >
                  GMM
                </a>
              </td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td className="emoji">&#10004;</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td className="module">
                <a href="https://github.com/microsoft/ADBench" target="_blank">
                  BA
                </a>
              </td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td className="emoji">&#8987;</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td className="module">
                <a href="https://github.com/microsoft/ADBench" target="_blank">
                  HT
                </a>
              </td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
            </tr>
          </table>
        </div>
        <div>
          <p className="text">
            This table displays the current supported tools and functions in
            GradBench. Our first eval is a simple{" "}
            <b style={{ color: "#3e4756" }}>Hello</b> module that computes the
            derivative of x&sup2;. <b style={{ color: "#3e4756" }}>Hello</b> can
            be run on any of the tools listed in the table.
            <br />
            <br />
            We are currently working on implementing functions from Microsoft's
            ADBench suite. The Gaussian Mixture Model Fitting (
            <b style={{ color: "#3e4756" }}>GMM</b>) equation is currently only
            supported by PyTorch. Additionally, PyTorch's support for the Bundle
            Adjustement (<b style={{ color: "#3e4756" }}>BA</b>) equation is
            in-progress.
          </p>
        </div>
      </div>
    </>
  );
};

export default App;
