use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::array::LargeListArray;
use arrow_array::builder::{LargeListBuilder, UInt32Builder};
use arrow_array::{Array, StringArray};
use arrow_data::ArrayData;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use tokenizers::tokenizer::Tokenizer;

#[pyclass(name = "LargeListArray")]
#[derive(Clone)]
pub struct PyArrowLargeListArray(LargeListArray);

impl From<PyArrowLargeListArray> for PyObject {
    fn from(val: PyArrowLargeListArray) -> Self {
        Python::with_gil(|py| val.0.to_data().to_pyarrow(py).unwrap())
    }
}

use jieba_rs::Jieba;
use std::collections::HashMap;
use std::io::BufReader;

#[pyclass]
struct ChineseTokenizer {
    engine: Jieba,
    word_to_id: HashMap<String, u32>,
}

#[pymethods]
impl ChineseTokenizer {
    #[new]
    fn new(dict_path: Option<&str>) -> PyResult<Self> {
        let mut engine = Jieba::new();
        if let Some(path) = dict_path {
            let file = std::fs::File::open(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to open dict file: {}", e))
            })?;
            let mut reader = BufReader::new(file);
            engine.load_dict(&mut reader).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to load dict: {}", e))
            })?;
        }
        Ok(ChineseTokenizer {
            engine,
            word_to_id: HashMap::new(),
        })
    }

    /// Sets the word-to-id mapping.
    fn set_vocab(&mut self, vocab: HashMap<String, u32>) {
        self.word_to_id = vocab;
    }

    /// Tokenizes a pyarrow StringArray and returns a pyarrow.LargeListArray.
    #[pyo3(signature = (texts))]
    fn tokenize(&self, texts: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Convert pyarrow object to rust StringArray
        let array_data = ArrayData::from_pyarrow_bound(texts)?;
        let array = arrow_array::make_array(array_data);
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("Expected a StringArray"))?;

        let results: PyResult<Vec<Option<Vec<u32>>>> = (0..string_array.len())
            .into_par_iter()
            .map(|i| {
                if string_array.is_null(i) {
                    return Ok(None);
                }
                let text = string_array.value(i);
                // Chinese segmentation using jieba
                let words = self.engine.cut(text, false); // No HMM for speed by default
                let mut ids = Vec::new();
                for word in words {
                    if let Some(&id) = self.word_to_id.get(word) {
                        ids.push(id);
                    }
                }
                Ok(Some(ids))
            })
            .collect();

        let results = results?;

        let values_capacity: usize = results
            .iter()
            .filter_map(|x| x.as_ref())
            .map(|x| x.len())
            .sum();
        let values_builder = UInt32Builder::with_capacity(values_capacity);
        let mut list_builder = LargeListBuilder::with_capacity(values_builder, string_array.len());

        for ids_opt in results {
            if let Some(ids) = ids_opt {
                list_builder.values().append_slice(&ids);
                list_builder.append(true);
            } else {
                list_builder.append(false);
            }
        }

        // Build the final LargeListArray.
        let list_array = list_builder.finish();
        Ok(PyArrowLargeListArray(list_array).into())
    }
}

#[pyclass]
struct ArrowTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl ArrowTokenizer {
    #[new]
    fn new(json_content: &str) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_bytes(json_content.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to load tokenizer from json: {}",
                e
            ))
        })?;
        Ok(ArrowTokenizer { tokenizer })
    }

    /// Serializes the tokenizer to a JSON string.
    fn to_str(&self) -> PyResult<String> {
        self.tokenizer
            .to_string(false) // Not using pretty print
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to serialize tokenizer to string: {}",
                    e
                ))
            })
    }

    /// Tokenizes a pyarrow StringArray and returns a pyarrow.LargeListArray.
    #[pyo3(signature = (texts, add_special_tokens = false))]
    fn tokenize(&self, texts: &Bound<'_, PyAny>, add_special_tokens: bool) -> PyResult<PyObject> {
        // Convert pyarrow object to rust StringArray
        let array_data = ArrayData::from_pyarrow_bound(texts)?;
        let array = arrow_array::make_array(array_data);
        let string_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("Expected a StringArray"))?;

        let results: PyResult<Vec<Option<Vec<u32>>>> = (0..string_array.len())
            .into_par_iter()
            .map(|i| {
                if string_array.is_null(i) {
                    return Ok(None);
                }
                let text = string_array.value(i);
                self.tokenizer
                    .encode_fast(text, add_special_tokens)
                    .map(|encoding| Some(encoding.get_ids().to_vec()))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to encode text: {}",
                            e
                        ))
                    })
            })
            .collect();

        let results = results?;

        let values_capacity: usize = results
            .iter()
            .filter_map(|x| x.as_ref())
            .map(|x| x.len())
            .sum();
        let values_builder = UInt32Builder::with_capacity(values_capacity);
        let mut list_builder = LargeListBuilder::with_capacity(values_builder, string_array.len());

        for ids_opt in results {
            if let Some(ids) = ids_opt {
                list_builder.values().append_slice(&ids);
                list_builder.append(true);
            } else {
                list_builder.append(false);
            }
        }

        // Build the final LargeListArray.
        let list_array = list_builder.finish();
        Ok(PyArrowLargeListArray(list_array).into())
    }
}

/// Python module implemented in Rust.
#[pymodule]
fn arrow_tokenize(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowTokenizer>()?;
    m.add_class::<ChineseTokenizer>()?;
    // Expose the crate version as `arrow_tokenize.__version__`
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_chinese_segmentation() {
        let tokenizer = ChineseTokenizer::new(None).unwrap();
        let words = tokenizer.engine.cut("我愛北京天安門", false);
        assert!(words.contains(&"我"));
        assert!(words.contains(&"愛"));
        assert!(words.contains(&"北京"));
    }

    #[test]
    fn test_custom_dict_loading() {
        let mut file = NamedTempFile::new().unwrap();
        // jieba format: word count tag
        writeln!(file, "高鐵假期 100 n").unwrap();
        
        let path = file.path().to_str().unwrap();
        let tokenizer = ChineseTokenizer::new(Some(path)).unwrap();
        let words = tokenizer.engine.cut("這是一個高鐵假期", false);
        assert!(words.contains(&"高鐵假期"));
    }
}
