import "./App.css";

const App = () => {
  return (
    <>
      <div className="header-cont">
        <a href="https://github.com/gradbench/gradbench" target="_blank">
          <h1 className="header">
            GradBench
            <span className="subtitle">
              A Benchmark for Differentiable Programming Across Languages and
              Domains
            </span>
            {/* <a className="repo" href="https://github.com/gradbench/gradbench">
            GitHub Repository
          </a> */}
          </h1>
        </a>
      </div>

      {/* <h2>Automatic Differentiation</h2>
      <p>
        Automatic Differentiation (AD) is the process of differentiation a
        mathematical function with respect to its inputs. Fields such as
        interactivity and machine learning benefit greatly when programs can
        automatically compute derivatives.{" "}
      </p> */}

      <div className="row">
        <h2 className="subheading">Benchmarking</h2>
        <div className="fillspace"></div>
      </div>
      <p className="text">
        GradBench is designed to work across languages and domains. Functions
        are written in <b style={{ color: "#3e4756" }}>Adroit</b> and
        automatically translated to a tool's native language. The architecture
        allows for the easy addition of new functions and tools.
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
          <p className="text">TODO: Explain Table</p>
        </div>
      </div>
    </>
  );
};

export default App;
