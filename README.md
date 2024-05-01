# knowledge graph facilitated drug response prediction




##### Step 1: Data processing
[Generate feature matrix for the training set Beat AML Wave 1/2](./notebooks/Step0_Generate_feature_matrix_beatAML_wave12.ipynb) <br>
[Generate feature matrix for the test data set Beat AML Wave 3/4](./notebooks/Step0_Generate_feature_matrix_beatAML_wave1_4.ipynb)<br>
[Generate feature matrix for the Functional Precision Medicine Tumor Board](./notebooks/Step0_Generate_feature_matrix_Finland.ipynb)<br>


##### Step 2: Construct a minimal graph to facilitate drug response prediction for Acute Myeloid Leukemia (AML)
[Extract TF miRNA to target gene pairs for the knowledge graph](./notebooks/Step1_Extract_TF_miRNA_to_targetGene_pairs.ipynb) <br>
[Fine house keeping genes as reference](./notebooks/Step2_Housekeeping_gene_selection_basedon_Variance_analysis.ipynb)

##### Step 3: Building predictive models
[Classifier for Venetoclax](./notebooks/Step5_compared_models_between_different_batches_Venetoclax.ipynb)<br>

##### Step 4: Feature exploration