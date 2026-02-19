use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::os::unix::io::AsRawFd;

/// Async batch loader using parallel pread
#[pyclass]
struct AsyncBatchLoader {
    fd: i32,
}

#[pymethods]
impl AsyncBatchLoader {
    #[new]
    fn new(file_path: String) -> PyResult<Self> {
        // Open file
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let fd = file.as_raw_fd();
        
        // On macOS, use F_NOCACHE instead of O_DIRECT
        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }
        
        std::mem::forget(file); // Keep fd alive
        
        Ok(AsyncBatchLoader { fd })
    }
    
    /// Load multiple chunks in parallel
    fn load_batch(&self, py: Python, requests: Vec<(u64, usize)>) -> PyResult<Vec<PyObject>> {
        use std::thread;
        
        py.allow_threads(|| {
            let fd = self.fd;
            let handles: Vec<_> = requests
                .into_iter()
                .map(|(offset, length)| {
                    thread::spawn(move || {
                        let mut buffer = vec![0u8; length];
                        let result = unsafe {
                            libc::pread(
                                fd,
                                buffer.as_mut_ptr() as *mut libc::c_void,
                                length,
                                offset as i64,
                            )
                        };
                        
                        if result < 0 {
                            Err(std::io::Error::last_os_error())
                        } else {
                            Ok(buffer)
                        }
                    })
                })
                .collect();
            
            let mut results = Vec::new();
            for handle in handles {
                let buffer = handle
                    .join()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Thread panicked"))?
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                
                Python::with_gil(|py| {
                    results.push(PyBytes::new(py, &buffer).into());
                });
            }
            
            Ok(results)
        })
    }
    
    fn close(&self) -> PyResult<()> {
        unsafe {
            libc::close(self.fd);
        }
        Ok(())
    }
}

/// SIMD-optimized dot product for f32
#[pyfunction]
fn dot_product_f32(py: Python, a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Vectors must have same length"));
    }
    
    py.allow_threads(|| {
        Ok(dot_product_simd(&a, &b))
    })
}

/// SIMD dot product implementation
#[inline]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON SIMD (Apple Silicon)
        unsafe { dot_product_neon(a, b) }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback to auto-vectorized code
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    
    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    
    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb);  // Fused multiply-add
        i += 4;
    }
    
    // Sum the vector
    let mut result = vaddvq_f32(sum);
    
    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

/// Batch dot product (query vs multiple vectors)
#[pyfunction]
fn batch_dot_product(py: Python, query: Vec<f32>, vectors: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    py.allow_threads(|| {
        Ok(vectors.iter().map(|v| dot_product_simd(&query, v)).collect())
    })
}

/// Optimized 4-bit distance computation (IP metric)
/// codes: flattened uint8 array (n_vectors * dim)
/// Returns negative dot products (distances)
#[pyfunction]
fn compute_4bit_distances_ip(
    py: Python,
    query: Vec<f32>,
    codes: Vec<u8>,
    n_vectors: usize,
    scale: Vec<f32>,
    offset: Vec<f32>,
) -> PyResult<Vec<f32>> {
    let dim = query.len();
    
    if codes.len() != n_vectors * dim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("codes length {} != n_vectors {} * dim {}", codes.len(), n_vectors, dim)
        ));
    }
    
    py.allow_threads(|| {
        let mut dists = vec![0.0f32; n_vectors];
        
        for i in 0..n_vectors {
            let code_start = i * dim;
            let code_slice = &codes[code_start..code_start + dim];
            
            // Compute: <query, codes*scale + offset>
            let mut dot = 0.0f32;
            for d in 0..dim {
                let reconstructed = code_slice[d] as f32 * scale[d] + offset[d];
                dot += query[d] * reconstructed;
            }
            dists[i] = -dot;  // Negate for distance (higher IP = lower distance)
        }
        
        Ok(dists)
    })
}

/// Python module
#[pymodule]
fn fast_io(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AsyncBatchLoader>()?;
    m.add_function(wrap_pyfunction!(dot_product_f32, m)?)?;
    m.add_function(wrap_pyfunction!(batch_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(compute_4bit_distances_ip, m)?)?;
    Ok(())
}
