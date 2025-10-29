## Snakemake Niceties
To visualize DAGs of the pipeline or generate a report, use the following commands:

```
# Visualizing the processing pipeline for a single execution
snakemake --configfile <your_config.yaml> --rulegraph | dot -Tpdf > dag_rulegraph_template.pdf

# Visualizing the graph for all pipeline executions
snakemake --configfile <your_config.yaml> --dag > dag.dot
dot -Tpdf dag.dot > dag_custom.pdf

# Generate a report for a finished pipeline execution (to set graph and runtimes)
snakemake --configfile <your_config.yaml> --report report.html


# Manually adjusting intermediate outputs
Sometimes you might want to manually adjust some intermediate outputs to see how this would affect the results.
For this you can go into the file, adjust some values or replace it with a new file with the same name.
Then you will need to delete all downstream rules of this file, to force snakemake to regenerate them.
Aditionally you will need to run snakemake with the --touch flag. This let's snakemake know, that even though a file was
altered, it is considered valid. You can then run the pipeline as usual with the intermediate output not being overwritten.
```

Other helpful command line args are documented on the [snakemake CLI page](https://snakemake.readthedocs.io/en/stable/executing/cli.html).




# Tests
Tests are handled by pytest. Navigate to the root directory and execute `pytest` to run them.
