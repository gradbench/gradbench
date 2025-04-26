use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize an optional JSON value as `Some`, so only missing values become `None`.
fn deserialize_optional_json<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error> {
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(Some(value))
}

/// A message ID.
pub type Id = i64;

/// A message from the eval.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Message {
    /// The first message.
    Start {
        /// The message ID.
        id: Id,

        /// The eval name.
        #[serde(
            default, // Deserialize as `None` if missing.
            skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
        )]
        eval: Option<String>,

        /// The optional eval configuration.
        #[serde(
            default, // Deserialize as `None` if missing.
            deserialize_with = "deserialize_optional_json", // Deserialize as `Some` if present.
            skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
        )]
        config: Option<serde_json::Value>,
    },

    /// A request to define a module.
    Define {
        /// The message ID.
        id: Id,

        /// The name of the module.
        module: String,
    },

    /// A request to evaluate a function.
    Evaluate {
        /// The message ID.
        id: Id,

        /// The name of the module.
        module: String,

        /// The name of the function.
        function: String,

        /// The input to the function.
        #[serde(
            default, // Deserialize as `None` if missing.
            deserialize_with = "deserialize_optional_json", // Deserialize as `Some` if present.
            skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
        )]
        input: Option<serde_json::Value>,

        /// An optional and short human-readable description of the input.
        #[serde(
            default, // Deserialize as `None` if missing.
            skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
        )]
        description: Option<String>,
    },

    /// Analysis results from evaluating a function.
    Analysis {
        /// The message ID.
        id: Id,

        /// The ID of the original message being analyzed.
        of: Id,

        /// Whether the tool's response was valid.
        valid: bool,

        /// An optional error message if the tool's response was invalid.
        #[serde(
            default, // Deserialize as `None` if missing.
            skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
        )]
        error: Option<String>,
    },
}

/// Nanosecond timings from the tool.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Timing {
    /// The name of this timing.
    pub name: String,

    /// How many nanoseconds elapsed in this timing.
    pub nanoseconds: u128,
}

/// A response from the tool to a `"start"` message.
#[derive(Debug, Deserialize, Serialize)]
pub struct StartResponse {
    /// The message ID.
    pub id: Id,

    /// The tool name.
    #[serde(
        default, // Deserialize as `None` if missing.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub tool: Option<String>,

    /// The optional tool configuration.
    #[serde(
        default, // Deserialize as `None` if missing.
        deserialize_with = "deserialize_optional_json", // Deserialize as `Some` if present.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub config: Option<serde_json::Value>,
}

/// A response from the tool to a `"define"` message.
#[derive(Debug, Deserialize, Serialize)]
pub struct DefineResponse {
    /// The message ID.
    pub id: Id,

    /// Whether the module was successfully defined.
    pub success: bool,

    /// Subtask timings.
    #[serde(
        default, // Deserialize as `None` if missing.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub timings: Option<Vec<Timing>>,

    /// An optional error message, if definition failed.
    #[serde(
        default, // Deserialize as `None` if missing.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub error: Option<String>,
}

/// A response from the tool to an `"evaluate"` message.
#[derive(Debug, Deserialize, Serialize)]
pub struct EvaluateResponse {
    /// The message ID.
    pub id: Id,

    /// Whether evaluation was successful.
    pub success: bool,

    /// The output of the function, if evaluation was successful.
    #[serde(
        default, // Deserialize as `None` if missing.
        deserialize_with = "deserialize_optional_json", // Deserialize as `Some` if present.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub output: Option<serde_json::Value>,

    /// Subtask timings.
    pub timings: Option<Vec<Timing>>,

    /// An optional error message, if evaluation failed.
    #[serde(
        default, // Deserialize as `None` if missing.
        skip_serializing_if = "Option::is_none" // Serialize as missing if `None`.
    )]
    pub error: Option<String>,
}

/// A response from the tool to an `"analysis"` message.
#[derive(Debug, Deserialize, Serialize)]
pub struct AnalysisResponse {
    /// The message ID.
    pub id: Id,
}

/// A nanoseconds object.
#[derive(Debug, Deserialize, Serialize)]
pub struct Nanoseconds {
    pub nanoseconds: u128,
}

/// A message entry in a log file.
#[derive(Debug, Deserialize, Serialize)]
pub struct LogMessage {
    /// The timestamp in nanoseconds.
    pub elapsed: Nanoseconds,

    /// The contained message.
    pub message: Message,
}

/// A response entry in a log file.
#[derive(Debug, Deserialize, Serialize)]
pub struct LogResponse<T> {
    /// The timestamp in nanoseconds.
    pub elapsed: Nanoseconds,

    /// The contained response.
    pub response: T,
}
