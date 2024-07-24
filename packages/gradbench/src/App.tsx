import "./App.css";

const App = () => {
  return (
    <>
      <a href="https://github.com/gradbench/gradbench" target="_blank">
        <h1 className="header">
          GradBench
          <span className="subtitle">
            A Benchmark for Differentiable Programming Across Languages and
            Domains
          </span>
        </h1>
      </a>

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
      <p className="text">
        <li> GradBench is designed to work across languages and domains. </li>
        <li>
          Functions are written in <b style={{ color: "#3e4756" }}>Adroit</b>{" "}
          and automatically translated to a tool's native language.
        </li>
        <li>
          The architecture allows for the easy addition of new functions and
          tools.
        </li>
      </p>

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
        <table>
          <tr>
            <th className="no-border"></th>
            <th className="tool">
              <a href="https://github.com/HIPS/autograd" target="_blank">
                Autograd
              </a>
            </th>
            <th className="tool">
              <a href="https://diffsharp.github.io/index.html" target="_blank">
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
