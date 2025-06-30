use pyo3::prelude::*;
use rand;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, ErrorKind};


#[pymodule]
fn cstgs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(number_of_triangles, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn number_of_triangles(
    addr: String,
    processors: u32,
    edge_sampling_prob: f64,
    wedge_sampling_prob: f64,
) -> io::Result<u64> {
    // Read all edges first to avoid concurrent file access
    let reader = Reader::new(&addr)?;
    let edges: Vec<_> = reader.collect::<io::Result<Vec<_>>>()?;

    // Split edges into chunks for parallel processing
    let chunk_size = (edges.len() as f64 / processors as f64).ceil() as usize;
    let chunks: Vec<_> = edges.chunks(chunk_size).collect();

    // Process chunks in parallel
    let counts: Vec<u64> = chunks
        .par_iter()
        .map(|chunk| {
            let mut g = CSTGS::new(edge_sampling_prob, wedge_sampling_prob)
                .expect("Failed to create CSTGS");
            let mut t: u64 = 0;
            for edge in chunk.iter() {
                t += g.add(edge.clone());
            }
            t
        })
        .collect();

    // Sum all counts and apply sampling correction
    let total: u64 = counts.iter().sum();
    Ok((total as f64 / (edge_sampling_prob * edge_sampling_prob + wedge_sampling_prob)) as u64)
}

struct Reader {
    reader: BufReader<File>,
    line_num: u64,
}

impl Reader {
    pub fn new(addr: &str) -> io::Result<Self> {
        let reader = BufReader::new(File::open(addr)?);
        Ok(Self {
            reader,
            line_num: 0,
        })
    }
}

impl Iterator for Reader {
    type Item = io::Result<Edge>;

    fn next(&mut self) -> Option<Self::Item> {
        self.line_num += 1;
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => match Edge::from_tsv(line.trim_end().to_string()) {
                Ok(edge) => Some(Ok(edge)),
                Err(error) => Some(Err(io::Error::new(
                    error.kind(),
                    format!("error occured on line {}; ", self.line_num) + &error.to_string(),
                ))),
            },
            Err(error) => Some(Err(error)),
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone)]
struct Edge(String, String);

impl Edge {
    fn new(v1: String, v2: String) -> Self {
        if v1 > v2 {
            return Self(v2, v1);
        }
        Self(v1, v2)
    }
    fn from_tsv(line: String) -> io::Result<Self> {
        let mut i: u64 = 0;
        let mut v1: String = String::new();
        let mut v2: String = String::new();
        for v in line.trim_end().split('\t') {
            if i == 0 {
                v1 = v.to_string();
            } else if i == 1 {
                v2 = v.to_string();
            } else {
                let err_msg =
                    format!("each line must contain two tab-separated numbers or strings");
                return Err(io::Error::new(ErrorKind::InvalidData, err_msg));
            }
            i += 1;
        }
        if v2.is_empty() {
            let err_msg = format!("each line must contain two tab-separated numbers or strings");
            return Err(io::Error::new(ErrorKind::InvalidData, err_msg));
        }
        Ok(Self::new(v1, v2))
    }
}

struct CSTGS {
    edge_sprob: f64,
    wedge_sprob: f64,
    edges: HashSet<Edge>,
    neighbours: HashMap<String, Vec<String>>,
}

impl CSTGS {
    fn new(edge_sampling_prob: f64, wedge_sampling_prob: f64) -> io::Result<Self> {
        if !CSTGS::is_valid_prob(edge_sampling_prob) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                String::from("edge_sampling_prob must be in [0, 1] interval"),
            ));
        }
        if !CSTGS::is_valid_prob(wedge_sampling_prob) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                String::from("wedge_sampling_prob must be in [0, 1] interval"),
            ));
        }
        Ok(Self {
            edge_sprob: edge_sampling_prob,
            wedge_sprob: wedge_sampling_prob,
            edges: HashSet::new(),
            neighbours: HashMap::new(),
        })
    }

    fn is_valid_prob(p: f64) -> bool {
        if p > 1.0 || p < 0.0 {
            return false;
        }
        return true;
    }

    fn add(&mut self, edge: Edge) -> u64 {
        let mut triangle_count = 0;

        // Step 1: Sample the edge with probability edge_sprob
        if Self::pass(self.edge_sprob) {
            let (u, v) = (&edge.0, &edge.1);

            // Add edge to edges set
            self.edges.insert(edge.clone());

            // Step 1: Check for wedges formed by the new edge
            // Check neighbors of u to form wedges (u, w), (w, v)
            if let Some(u_neighbors) = self.neighbours.get(u) {
                for w in u_neighbors {
                    if w != v {
                        let potential_edge = Edge::new(w.clone(), v.clone());
                        if self.edges.contains(&potential_edge) && Self::pass(self.wedge_sprob) {
                            // Wedge (u, w), (w, v) is sampled
                            // No need to store wedge explicitly; we can use it later if needed
                        }
                    }
                }
            }
            // Check neighbors of v to form wedges (v, w), (w, u)
            if let Some(v_neighbors) = self.neighbours.get(v) {
                for w in v_neighbors {
                    if w != u {
                        let potential_edge = Edge::new(w.clone(), u.clone());
                        if self.edges.contains(&potential_edge) && Self::pass(self.wedge_sprob) {
                            // Wedge (v, w), (w, u) is sampled
                        }
                    }
                }
            }
            // Update neighbors for u and v
            self.neighbours
                .entry(u.clone())
                .or_insert_with(|| Vec::new())
                .push(v.clone());
            self.neighbours
                .entry(v.clone())
                .or_insert_with(|| Vec::new())
                .push(u.clone());
        }

        // Step 2: Check if the edge closes any wedges to form triangles
        let (u, v) = (&edge.0, &edge.1);
        if let Some(u_neighbors) = self.neighbours.get(u) {
            if let Some(v_neighbors) = self.neighbours.get(v) {
                // Find common neighbors w where (u, w) and (v, w) exist
                for w in u_neighbors {
                    if v_neighbors.contains(w) && w != u && w != v {
                        // Check if both edges (u, w) and (v, w) were sampled
                        let edge_uw = Edge::new(u.clone(), w.clone());
                        let edge_vw = Edge::new(v.clone(), w.clone());
                        if self.edges.contains(&edge_uw) && self.edges.contains(&edge_vw) {
                            // Since we're not storing wedges explicitly, apply wedge sampling here
                            if Self::pass(self.wedge_sprob) {
                                triangle_count += 1;
                            }
                        }
                    }
                }
            }
        }

        triangle_count
    }

    fn pass(p: f64) -> bool {
        if rand::random::<f64>() < p {
            return true;
        }
        return false;
    }
}
