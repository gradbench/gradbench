import "./App.css";

const App = () => {
  return (
    <>
      <div className="header-cont">
        <h1 className="header">
          GradBench
          <span className="subtitle">
            An Automatic Differentiation Benchmarking Suite
          </span>
          {/* <a className="repo" href="https://github.com/gradbench/gradbench">
            GitHub Repository
          </a> */}
        </h1>
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
        are written in <b>Adroit</b> and automatically translated a to tools
        native language. The architexture allows for the easy addition of new
        functions and tools.
      </p>

      <div className="row">
        <h2 className="subheading">Adroit</h2>
        <div className="fillspace"></div>
      </div>

      <p className="text">Explain adroit</p>

      <div className="row">
        <h2 className="subheading">Currently Implemented</h2>
        <div className="fillspace"></div>
      </div>

      <div className="container">
        <table>
          <tr>
            <th className="no-border"></th>
            <th className="tool">Autograd</th>
            <th className="tool">DiffSharp</th>
            <th className="tool">Jax</th>
            <th className="tool">MyGrad</th>
            <th className="tool">PyTorch</th>
            <th className="tool">SciLean</th>
            <th className="tool">Tapenade</th>
            <th className="tool">TensorFlow</th>
            <th className="tool">Zygote</th>
          </tr>
          <tr>
            <td className="module">Hello</td>
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
            <td className="module">GMM</td>
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
            <td className="module">BA</td>
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
            <td className="module">HT</td>
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
          <p className="text">Table explanation</p>
        </div>
      </div>

      <div className="row">
        <h2 className="subheading">Links</h2>
        <div className="fillspace"></div>
      </div>
      <div className="container">
        <div className="text">
          <h3 className="subsub">Tools:</h3>
          <a href="https://github.com/HIPS/autograd">Autograd</a> <br />
          <a href="https://diffsharp.github.io/index.html">DiffSharp</a>
          <br />
          <a href="https://jax.readthedocs.io/en/latest/index.html">Jax</a>
          <br />
          <a href="https://mygrad.readthedocs.io/en/latest/">MyGrad</a>
          <br />
          <a href="https://pytorch.org/">PyTorch</a>
          <br />
          <a href="https://github.com/lecopivo/SciLean">SciLean</a>
          <br />
          <a href="https://tapenade.gitlabpages.inria.fr/userdoc/build/html/index.html">
            Tapenade
          </a>
          <br />
          <a href="https://www.tensorflow.org/">TensorFlow</a>
          <br />
          <a href="https://fluxml.ai/Zygote.jl/stable/">Zygote</a>
          <br />
        </div>
        <div className="text">
          <h3 className="subsub">Benchmarks:</h3>
          <a href="https://github.com/microsoft/ADBench">ADBench</a> <br />
        </div>
      </div>
    </>
  );
};

export default App;
