- Train

  - The WikiSQL dataset has been preprocessed in the data folder (graph format & placeholder replacement).

  - Use the following shell to train.

    ```shell
    cd main
    PYTHONIOENCODING=utf-8 python3 -u run_model.py train -sample_layer_size 2 -epochs 100
    ```

- Augment

  - Sample SQL from db. Just follow the code in table2sql.ipynb.

  - Preprocess the sampled data into the format model needs. Just follow the code in sql2graph.ipynb.

  - Inference. Make sure you have move the file generated by the last step to the data/no_cycle folder and rename it as test.data. Use the following shell to inference and the output will be placed in saved_model folder.

    ```shell
    PYTHONIOENCODING=utf-8 python3 -u run_model.py test -sample_layer_size 2
    ```

  - Finally, follow the code in postprocess.ipynb to generate the augmented data in WikiSQL format. You can use any NL2SQL model to validate the performance of these augmented data.

