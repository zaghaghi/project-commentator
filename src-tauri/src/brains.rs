// The "brain" catalog: the QAT GGUF build of gemma-4-E2B plus its vision
// projector, downloaded from HuggingFace on demand. Each brain is two files —
// the text LM (the QAT main model) and the mmproj vision projector that lets
// it see the screenshots we hand it.

use std::path::PathBuf;

use serde::Serialize;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Brain {
    pub id: &'static str,
    pub label: &'static str,
    pub blurb: &'static str,
    #[serde(skip)]
    pub repo: &'static str,
    pub main_file: &'static str,
    pub mmproj_file: &'static str,
    pub main_size_bytes: u64,
    pub mmproj_size_bytes: u64,
}

/// The only brain, auto-loaded on startup.
pub const ACTIVE: &str = "gemma-4-e2b";

pub const CATALOG: &[Brain] = &[Brain {
    id: "gemma-4-e2b",
    label: "Gemma 4 E2B QAT",
    blurb: "A small, sharp Gemma 4 with vision. Downloads the QAT text model plus its mmproj projector on first run.",
    repo: "google/gemma-4-E2B-it-qat-q4_0-gguf",
    main_file: "gemma-4-E2B_q4_0-it.gguf",
    mmproj_file: "gemma-4-E2B-it-mmproj.gguf",
    main_size_bytes: 3_350_000_000,
    mmproj_size_bytes: 1_280_000_000,
}];

pub fn find(id: &str) -> Option<&'static Brain> {
    CATALOG.iter().find(|b| b.id == id)
}

pub fn active() -> &'static Brain {
    find(ACTIVE).expect("ACTIVE brain missing from CATALOG")
}

/// Where downloaded brains are cached (persists across runs).
pub fn cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".project-commentator")
        .join("brains")
}

impl Brain {
    pub fn main_path(&self) -> PathBuf {
        cache_dir().join(self.main_file)
    }
    pub fn mmproj_path(&self) -> PathBuf {
        cache_dir().join(self.mmproj_file)
    }
    /// Both files must be present before we can load.
    pub fn is_downloaded(&self) -> bool {
        self.main_path().exists() && self.mmproj_path().exists()
    }
    pub fn resolve_url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, file)
    }
}