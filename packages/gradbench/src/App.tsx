import "./App.css";
import logo from "./logo_t.svg";

const App = () => {
  return (
    <>
      <div className="stick">
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
              color: "white",
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
      </div>
      <h1 className="header">
        <img src={logo}></img>
        GradBench
        <span className="subtitle">
          A Benchmark for Differentiable Programming Across Languages and
          Domains
        </span>
      </h1>

      {/* Appears for mobile view only */}
      <p className="hidden">
        <span className="bold">
          A Benchmark for Differentiable Programming Across Languages and
          Domains
        </span>
      </p>

      <div className="subheader-container">
        <h2 className="subheading">Differentiable Programming</h2>
        <div className="box">
          <p className="text">
            Differentiable programming is an approach that leverages{" "}
            <span className="bold">automatic differentiation</span> (autodiff).
            It involves programs capable of automatically computing the
            derivatives of mathematical functions with respect to their inputs.
            This capability is particularly useful for
            <ul style={{ padding: "0.5em 1.5em" }}>
              <li>implementing interactivity in programs,</li>
              <li>training machine learning models with gradient descent,</li>
              <li>and solving optimization problems.</li>
            </ul>
            Overall, differentiable programming is a powerful tool that leads to
            more effective and adaptive code.
          </p>
        </div>
      </div>

      <div className="subheader-container">
        <h2 className="subheading">Benchmarking</h2>
        <div className="box">
          <div className="text">
            GradBench is a benchmarking suite for autodiff tools. It is designed
            to work across languages and domains. This is possible because
            functions are written in <span className="bold">Adroit</span> and
            automatically translated to a given tool's native language. This
            architecture allows for the easy addition of new functions and
            tools.
          </div>
        </div>
      </div>

      <div className="subheader-container">
        <h2 className="subheading">Adroit</h2>
        <div className="box">
          <div className="text">
            <span className="bold">Adroit</span> is a new differentiable
            programming language designed to be
            <ul style={{ padding: "0.5em 1.5em" }}>
              <li>
                flexible enough to easily express many numerical functions
                (whether involving arrays, recursion, higher-order functions,
                nested derivatives, custom derivatives, etc.), but also
              </li>
              <li>
                simple enough to mechanically translate from Adroit source code
                to a variety of different existing tools for automatic
                differentiation.
              </li>
            </ul>
            As an example, here is how you might define matrix multiplication in
            Adroit:
            <div
              style={{
                padding: "1em 2em",
              }}
            >
              <pre style={{ margin: 0, lineHeight: "125%" }}>
                <span style={{ color: "#008800", fontWeight: "bold" }}>
                  def
                </span>{" "}
                mmul[
                <span style={{ color: "#003366", fontWeight: "bold" }}>
                  M
                </span>,{" "}
                <span style={{ color: "#003366", fontWeight: "bold" }}>N</span>,{" "}
                <span style={{ color: "#003366", fontWeight: "bold" }}>P</span>
                ]({"\n"}
                {"  "}
                <span style={{ color: "#AA6600" }}>a:</span> [
                <span style={{ color: "#003366", fontWeight: "bold" }}>M</span>{" "}
                <span style={{ color: "#333333" }}>*</span>{" "}
                <span style={{ color: "#003366", fontWeight: "bold" }}>N</span>]
                <span style={{ color: "#003366", fontWeight: "bold" }}>
                  Float
                </span>
                ,{"\n"}
                {"  "}
                <span style={{ color: "#AA6600" }}>b:</span> [
                <span style={{ color: "#003366", fontWeight: "bold" }}>N</span>{" "}
                <span style={{ color: "#333333" }}>*</span>{" "}
                <span style={{ color: "#003366", fontWeight: "bold" }}>P</span>]
                <span style={{ color: "#003366", fontWeight: "bold" }}>
                  Float
                </span>
                {"\n"}): [
                <span style={{ color: "#003366", fontWeight: "bold" }}>M</span>{" "}
                <span style={{ color: "#333333" }}>*</span>{" "}
                <span style={{ color: "#003366", fontWeight: "bold" }}>P</span>]
                <span style={{ color: "#003366", fontWeight: "bold" }}>
                  Float
                </span>{" "}
                <span style={{ color: "#333333" }}>=</span>
                {"\n"}
                {"  "}for (i, j) <span style={{ color: "#333333" }}>=&gt;</span>
                {"\n"}
                {"    "}sum(for k{" "}
                <span style={{ color: "#333333" }}>=&gt;</span> a[i, k]{" "}
                <span style={{ color: "#333333" }}>*</span> b[k, j]){"\n"}
              </pre>
            </div>
            As shown in this example, Adroit supports typed indexes for arrays
            similar to{" "}
            <a href="https://futhark-lang.org/" target="_blank">
              Futhark
            </a>
            , and those index types can be algebraic types (similar to tuples or
            enums in many languages) as in{" "}
            <a
              href="https://github.com/google-research/dex-lang"
              target="_blank"
            >
              Dex
            </a>
            . The Adroit compiler is <span className="bold">fast</span>: it can
            lex, parse, and typecheck over 350k lines of code per second using
            one core on a modern laptop, causing little overhead as an initial
            preprocessing step before passing typed IR as JSON to another tool
            to perform automatic differentiation.
          </div>
        </div>
      </div>
      <div className="subheader-container">
        <h2 className="subheading">Currently Implemented</h2>
        <div className="container">
          <div className="box">
            <p className="text">
              This table displays the current supported tools and functions in
              GradBench. Our first eval is a simple{" "}
              <span className="bold">Hello</span> module that computes the
              derivative of x&sup2;. <span className="bold">Hello</span> can be
              run on any of the tools listed in the table.
              <br />
              <br />
              We are currently working on implementing functions from
              Microsoft's{" "}
              <a href="https://github.com/microsoft/ADBench" target="_blank">
                ADBench
              </a>{" "}
              suite. The{" "}
              <span className="bold">Gaussian Mixture Model Fitting</span> (GMM)
              and <span className="bold">Bundle Adjustment</span> (BA) equations
              are currently only supported by PyTorch. We have yet to implement
              the <span className="bold">Hand Tracking</span> (HT) equation.
            </p>
          </div>
          <div className="tbody">
            <table>
              <tr>
                <th className="no-border"></th>
                <th className="no-border">
                  <a href="https://github.com/HIPS/autograd" target="_blank">
                    <button className="tool">Autograd</button>
                  </a>
                </th>
                <th className="no-border">
                  <a
                    href="https://diffsharp.github.io/index.html"
                    target="_blank"
                  >
                    <button className="tool">DiffSharp</button>
                  </a>
                </th>
                <th className="no-border">
                  <a
                    href="https://jax.readthedocs.io/en/latest/index.html"
                    target="_blank"
                  >
                    <button className="tool">JAX</button>
                  </a>
                </th>
                <th className="no-border">
                  <a
                    href="https://mygrad.readthedocs.io/en/latest/"
                    target="_blank"
                  >
                    <button className="tool">MyGrad</button>
                  </a>
                </th>
                <th className="no-border">
                  <a href="https://pytorch.org/" target="_blank">
                    <button className="tool">PyTorch</button>
                  </a>
                </th>
                <th className="no-border">
                  <a href="https://github.com/lecopivo/SciLean" target="_blank">
                    <button className="tool">SciLean</button>
                  </a>
                </th>
                <th className="no-border">
                  <a
                    href="https://tapenade.gitlabpages.inria.fr/userdoc/build/html/index.html"
                    target="_blank"
                  >
                    <button className="tool">Tapenade</button>
                  </a>
                </th>
                <th className="no-border">
                  <a href="https://www.tensorflow.org/" target="_blank">
                    <button className="tool">TensorFlow</button>
                  </a>
                </th>
                <th className="no-border">
                  <a href="https://fluxml.ai/Zygote.jl/stable/" target="_blank">
                    <button className="tool">Zygote</button>
                  </a>
                </th>
              </tr>
              <tr>
                <td className="no-border">
                  {" "}
                  <a
                    href="https://github.com/gradbench/gradbench/tree/main/evals/hello"
                    target="_blank"
                  >
                    <button className="tool">Hello</button>
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
                <td className="no-border">
                  <a
                    href="https://github.com/gradbench/gradbench/tree/main/evals/gmm"
                    target="_blank"
                  >
                    <button className="tool">GMM</button>
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
                <td className="no-border">
                  <a
                    href="https://github.com/gradbench/gradbench/tree/main/evals/ba"
                    target="_blank"
                  >
                    <button className="tool">BA</button>
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
                <td className="no-border">
                  <a
                    href="https://github.com/microsoft/ADBench"
                    target="_blank"
                  >
                    <button className="tool">HT</button>
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
        </div>
      </div>
    </>
  );
};

export default App;
