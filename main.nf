#!/usr/bin/env nextflow
/*
 * ESM-C 600M vs ESM-2 650M: embed every input sequence with both models, one model on the
 * GPU at a time, then compare the two embedding spaces.
 *
 * Launched by run_server.sh, which builds the inputs and passes the env prefixes. Each task
 * calls its env's python by absolute path, so no conda activation happens inside tasks.
 * The Python scripts are staged as inputs so that editing one invalidates -resume caching.
 */

process EMBED {
    tag "${model}"
    maxForks 1  // one model on the GPU at a time: no memory contention, clean per-task trace timings
    publishDir "${params.outdir}/embeddings", mode: 'copy'

    input:
    tuple val(model), val(env_prefix)
    path fasta
    path embed_py

    output:
    path "${model}", emit: emb_dir

    script:
    """
    "${env_prefix}/bin/python" ${embed_py} \\
        --model ${model} \\
        --fasta ${fasta} \\
        --device ${params.device} \\
        --precision ${params.precision} \\
        --max_tokens ${params.max_tokens} \\
        --outdir ${model}
    """
}

process COMPARE {
    publishDir params.outdir, mode: 'copy'

    input:
    path emb_dirs  // esm2_650m/ and esmc_600m/, staged under their own names
    path manifest
    path fasta
    path compare_py

    output:
    path "comparison"

    script:
    """
    "${params.core_prefix}/bin/python" ${compare_py} \\
        --esm2 esm2_650m \\
        --esmc esmc_600m \\
        --manifest ${manifest} \\
        --fasta ${fasta} \\
        --perms ${params.perms} \\
        --outdir comparison
    """
}

workflow {
    ['input', 'manifest', 'core_prefix', 'esm2_prefix', 'esmc_prefix'].each { name ->
        if (!params[name]) {
            error "Missing --${name}. Launch the pipeline through run_server.sh, which sets it."
        }
    }
    fasta = file(params.input, checkIfExists: true)
    manifest = file(params.manifest, checkIfExists: true)

    models = channel.of(
        ['esm2_650m', params.esm2_prefix],
        ['esmc_600m', params.esmc_prefix]
    )
    EMBED(models, fasta, file("${projectDir}/bin/embed.py", checkIfExists: true))
    COMPARE(
        EMBED.out.emb_dir.collect(),
        manifest,
        fasta,
        file("${projectDir}/bin/compare_embeddings.py", checkIfExists: true)
    )
}
