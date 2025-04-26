# quiver_subspace

This projects provides an implementation of the generation of a subspace of a tensor product of irreducible representations of sl_n from a generating set by the action of the Iwahory algebra.

## 📁 Project Structure

├── gt_patterns.py # Core algorithm implementation 
├── utils.py # Utility/helper functions 
├── example.py # Example script using the algorithm 
├── requirements.txt # Python dependencies

## 🧪 Example Explained

The `example.py` file demonstrates how to use the algorithm implemented in `gt_patterns.py`. Here's a breakdown of what it does:

1. It imports the necessary functions from `alg.py` and `utils.py`.
2. It sets up lambda_mat, which is list of lists with integer elements such that the row i corresponds to the highest weight of the representation we want to put in the vertex i.
3. It sets up gen_set, which is the set of all extremal vectors. Any other lineraly independent generating set can be used.
4. It computes the subspace generated from gen_set and returns it and its dimension.
5. It computes the expected dimension and prints it as well as the dimension computed before. We expect them to be equal.

## 🚀 Getting Started

Follow the steps below to set up and run the project.

### 1. Clone the Repository

git clone https://github.com/AndreyK19/quiver_subspace.git
cd quiver_subspace

### 2. Create and Activate a Virtual Environment
python -m venv .venv
.venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt
