use tvm_vm::executor::zk_stuff::bn254::poseidon::PoseidonSponge;

pub struct PoseidonHasher {
    sponge: PoseidonSponge,
}

impl PoseidonHasher {
    pub fn new() -> Self {
        PoseidonHasher { sponge: PoseidonSponge::new() }
    }

    pub fn digest(&self, bytes: &[u8]) -> [u8; 32] {
        self.sponge.hash_bytes_flat(bytes).expect("Poseidon hash failed")
    }
}

impl Default for PoseidonHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl monotree::Hasher for PoseidonHasher {
    fn new() -> Self {
        PoseidonHasher::new()
    }

    fn digest(&self, bytes: &[u8]) -> [u8; 32] {
        self.digest(bytes)
    }
}

/// Hash a key-value pair into a leaf.
pub fn dense_leaf_hash(hasher: &PoseidonHasher, key: &[u8; 32], value: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(key);
    buf[32..].copy_from_slice(value);
    hasher.digest(&buf)
}

/// Hash two child nodes into a parent.
pub fn dense_combine(hasher: &PoseidonHasher, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    hasher.digest(&buf)
}

/// Build a dense Merkle tree in heap layout: root at index 0,
/// children of node `i` at `2i+1` (left) and `2i+2` (right).
/// Leaves occupy indices `[width-1 .. 2*width-2]`.
/// Returns the full tree array of size `2*width - 1`.
pub fn dense_merkle_tree(hasher: &PoseidonHasher, leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
    assert!(!leaves.is_empty());
    let width = leaves.len().next_power_of_two();
    let tree_size = 2 * width - 1;
    let mut tree = vec![[0u8; 32]; tree_size];

    // Fill leaves at the end of the array
    let leaf_start = width - 1;
    for (i, leaf) in leaves.iter().enumerate() {
        tree[leaf_start + i] = *leaf;
    }
    // Remaining slots are zero-padded

    // Build internal nodes bottom-up
    for i in (0..leaf_start).rev() {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        tree[i] = dense_combine(hasher, &tree[left], &tree[right]);
    }

    tree
}

/// Compute only the root hash (tree[0]).
pub fn dense_merkle_root(hasher: &PoseidonHasher, leaves: &[[u8; 32]]) -> [u8; 32] {
    dense_merkle_tree(hasher, leaves)[0]
}

/// Generate a Merkle proof for the leaf at position `pos`.
/// Returns sibling hashes from leaf level up to root.
pub fn dense_merkle_proof(
    hasher: &PoseidonHasher,
    leaves: &[[u8; 32]],
    pos: usize,
) -> Vec<[u8; 32]> {
    let tree = dense_merkle_tree(hasher, leaves);
    let width = leaves.len().next_power_of_two();
    let mut idx = width - 1 + pos;
    let mut proof = Vec::new();

    while idx > 0 {
        let sibling = if idx % 2 == 1 { idx + 1 } else { idx - 1 };
        proof.push(tree[sibling]);
        idx = (idx - 1) / 2;
    }

    proof
}

/// Verify a Merkle proof against an expected root.
pub fn dense_merkle_verify(
    hasher: &PoseidonHasher,
    root: &[u8; 32],
    leaf: &[u8; 32],
    pos: usize,
    proof: &[[u8; 32]],
) -> bool {
    let width = 1usize << proof.len();
    let mut idx = width - 1 + pos;
    let mut current = *leaf;

    for sibling in proof {
        current = if idx % 2 == 1 {
            dense_combine(hasher, &current, sibling)
        } else {
            dense_combine(hasher, sibling, &current)
        };
        idx = (idx - 1) / 2;
    }

    current == *root
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::time::Instant;

    fn random_hash(rng: &mut impl Rng) -> [u8; 32] {
        let mut h = [0u8; 32];
        rng.fill(&mut h);
        h
    }

    #[test]
    fn test_tree_128_leaves() {
        let mut rng = rand::thread_rng();
        const N: usize = 128;

        // Generate 128 random key-value pairs
        let keys: Vec<[u8; 32]> = (0..N).map(|_| random_hash(&mut rng)).collect();
        let values: Vec<[u8; 32]> = (0..N).map(|_| random_hash(&mut rng)).collect();

        // ===== Dense Balanced Tree (Poseidon) =====
        println!("\n===== Dense Balanced Tree (Poseidon) =====");

        // --- Hasher creation ---
        let t0 = Instant::now();
        let hasher = PoseidonHasher::new();
        let hasher_time = t0.elapsed();
        println!("PoseidonHasher::new()        : {:?}", hasher_time);

        // --- Leaf hashing ---
        let t1 = Instant::now();
        let leaves: Vec<[u8; 32]> = keys
            .iter()
            .zip(values.iter())
            .map(|(k, v)| dense_leaf_hash(&hasher, k, v))
            .collect();
        let leaf_time = t1.elapsed();
        println!("Hash {} leaves               : {:?}", N, leaf_time);

        // --- Tree construction ---
        let t2 = Instant::now();
        let tree = dense_merkle_tree(&hasher, &leaves);
        let tree_time = t2.elapsed();
        let root = tree[0];
        let dense_total = hasher_time + leaf_time + tree_time;
        println!("Build tree ({} leaves)       : {:?}", N, tree_time);
        println!("Tree size (nodes)            : {}", tree.len());
        println!("Root hash                    : {}", hex::encode(root));
        println!("Total (hasher+leaves+tree)   : {:?}", dense_total);

        // Verify tree structure
        assert_eq!(tree.len(), 2 * N - 1); // 128 is already a power of 2
        assert_ne!(root, [0u8; 32]);

        // ===== Monotree (Poseidon, MemoryDB) =====
        println!("\n===== Monotree (Poseidon, MemoryDB) =====");

        let t3 = Instant::now();
        let mut mono = monotree::Monotree::<monotree::database::MemoryDB, PoseidonHasher>::new("monotree");
        let mono_init_time = t3.elapsed();
        println!("Monotree::new()              : {:?}", mono_init_time);

        // --- Batch insert all 128 key-value pairs ---
        let t4 = Instant::now();
        let mono_root = mono.inserts(None, &keys, &values).expect("monotree inserts failed");
        let mono_insert_time = t4.elapsed();
        println!("Insert {} entries (batch)    : {:?}", N, mono_insert_time);
        println!("Root hash                    : {}", mono_root.map(hex::encode).unwrap_or_else(|| "None".into()));

        let mono_total = mono_init_time + mono_insert_time;
        println!("Total (init+inserts)         : {:?}", mono_total);

        // Verify all entries are retrievable
        let results = mono.gets(mono_root.as_ref(), &keys).expect("monotree gets failed");
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.as_ref(), Some(&values[i]), "monotree: mismatch at index {}", i);
        }
        println!("All {} entries verified       : OK", N);

        // ===== Comparison =====
        println!("\n===== Comparison =====");
        println!("Dense balanced (Poseidon)    : {:?}", dense_total);
        println!("Monotree (Poseidon)          : {:?}", mono_total);
        if dense_total > mono_total {
            let ratio = dense_total.as_nanos() as f64 / mono_total.as_nanos() as f64;
            println!("Monotree is {:.1}x faster", ratio);
        } else {
            let ratio = mono_total.as_nanos() as f64 / dense_total.as_nanos() as f64;
            println!("Dense balanced is {:.1}x faster", ratio);
        }
    }

    #[test]
    fn test_merkle_proof_128_leaves() {
        let mut rng = rand::thread_rng();
        const N: usize = 128;

        let hasher = PoseidonHasher::new();

        let keys: Vec<[u8; 32]> = (0..N).map(|_| random_hash(&mut rng)).collect();
        let values: Vec<[u8; 32]> = (0..N).map(|_| random_hash(&mut rng)).collect();

        let leaves: Vec<[u8; 32]> = keys
            .iter()
            .zip(values.iter())
            .map(|(k, v)| dense_leaf_hash(&hasher, k, v))
            .collect();

        let root = dense_merkle_root(&hasher, &leaves);

        // --- Proof generation & verification for every leaf ---
        let t_total = Instant::now();
        for pos in 0..N {
            let t_proof = Instant::now();
            let proof = dense_merkle_proof(&hasher, &leaves, pos);
            let proof_time = t_proof.elapsed();

            let t_verify = Instant::now();
            let ok = dense_merkle_verify(&hasher, &root, &leaves[pos], pos, &proof);
            let verify_time = t_verify.elapsed();

            assert!(ok, "Proof failed for leaf at pos {}", pos);
            assert_eq!(proof.len(), 7); // log2(128) = 7

            if pos == 0 || pos == 63 || pos == 127 {
                println!(
                    "Leaf {:>3}: proof_gen={:?}, verify={:?}, proof_len={}",
                    pos, proof_time, verify_time, proof.len()
                );
            }
        }
        let total_time = t_total.elapsed();
        println!(
            "All {} proofs generated & verified: {:?} (avg {:?}/proof)",
            N,
            total_time,
            total_time / N as u32
        );
    }

    #[test]
    fn test_invalid_proof_rejected() {
        let mut rng = rand::thread_rng();
        let hasher = PoseidonHasher::new();

        let leaves: Vec<[u8; 32]> = (0..128).map(|_| random_hash(&mut rng)).collect();
        let root = dense_merkle_root(&hasher, &leaves);

        // Correct proof for pos 0
        let proof = dense_merkle_proof(&hasher, &leaves, 0);
        assert!(dense_merkle_verify(&hasher, &root, &leaves[0], 0, &proof));

        // Wrong leaf should fail
        let wrong_leaf = random_hash(&mut rng);
        assert!(!dense_merkle_verify(&hasher, &root, &wrong_leaf, 0, &proof));

        // Wrong position should fail
        assert!(!dense_merkle_verify(&hasher, &root, &leaves[0], 1, &proof));

        // Tampered proof should fail
        let mut bad_proof = proof.clone();
        bad_proof[0] = random_hash(&mut rng);
        assert!(!dense_merkle_verify(&hasher, &root, &leaves[0], 0, &bad_proof));
    }

    #[test]
    fn test_deterministic_root() {
        let hasher = PoseidonHasher::new();

        // Fixed leaves
        let leaves: Vec<[u8; 32]> = (0u8..128)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0] = i;
                h
            })
            .collect();

        let root1 = dense_merkle_root(&hasher, &leaves);
        let root2 = dense_merkle_root(&hasher, &leaves);
        assert_eq!(root1, root2, "Same leaves must produce same root");

        // Different leaf → different root
        let mut leaves2 = leaves.clone();
        leaves2[0][0] = 0xFF;
        let root3 = dense_merkle_root(&hasher, &leaves2);
        assert_ne!(root1, root3, "Different leaves must produce different root");
    }
}
