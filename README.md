# ALGen-2

## New Features

- 🔬 **Substructural molecular filtering as a first inner Active Learning (AL) step:**  
  Incorporates substructural molecular filters that remove unwanted motifs before applying molecular property filters.
  These substructural filters include: XXX

- ⚖️ **Molecular weight added to molecular properties filter:**  
  In addition to Tanimoto similarity, QED, and SA, a **molecular weight range filter** has been added, allowing users to define both minimum and maximum MW limits.
  This is part of the second filter of the inner AL (molecular properties)

- 🔁 **Reversible Tanimoto similarity direction:**  
  A new inverted mode (`">"`) keeps molecules if **at least one tanimoto similarity value exceeds** the threshold. This is useful for lead optimization purposes, were we aim to generate molecules with small variations.

  The regular mode (`"<"`) continues to filter molecules **below** the maximum tanimoto threshold. This mode is useful for promoting diversity/variability.

- 📦 **Catalog filter integration in inner AL:**  
  Users can now enable RDKit **catalog filters** as an optional third filtering stage in the inner AL cycle. These catalogs are: XXX

- 🎯 **Subset-based similarity filtering (optional):**  
  Tanimoto similarity can now be computed against a **user-defined subset** instead of the specific set, enabling guided molecule generation toward user specific chemotypes.

- 🌿 **Trifurcated molecule filter (optional):**  
  Optionally detect and select **trifurcated structures**, with configurable minimum branch counts, for specialized design workflows.

- 🎲 **Fixed seed:**  
  Ensures reproducibility.

- 🔥 **Configurable sampling method:**  
  Users may choose between the original **temperature sampling** method or the newly added **top-p (nucleus) sampling** for more controlled generation.

- 🏋️ **Improved training configuration:**  
  Training now supports:
  - **CosineDecayRestarts** learning-rate schedule  
  - **Gradient clipping** in Adam  
  - **ReduceLROnPlateau** for adaptive learning-rate tuning  
  These updates improve stability and convergence on general datasets.

- 🧱 **(FUTURE IMPLEMENTATION) Scaffold-level similarity filtering:**  
  Future versions may compute Tanimoto similarity based on **scaffolds rather than full molecules**, enabling greater scaffold diversity during generation.

