#!/usr/bin/env nextflow
/*
 * ESM-C 600M vs ESM-2 650M: embed every input sequence with both models, one task on the
 * GPU at a time, then compare the two embedding spaces.
 *
 * EMBED runs once per (model, shard), so -resume keeps every finished shard when a long run
 * is interrupted (e.g. by a SLURM time limit); MERGE joins each model's shards.
 *
 * Launched by run_server.sh, which builds the inputs and passes the env prefixes. Each task
 * calls its env's python by absolute path, so no conda activation happens inside tasks.
 * The Python scripts are staged as inputs so that editing one invalidates -resume caching.
 */

process EMBED {
    tag "${model} ${shard.baseName}"
    maxForks 1  // one task on the GPU at a time: no memory contention, clean per-task timings

    input:
    tuple val(model), val(env_prefix), path(shard)
    path residue_sample
    path embed_py

    output:
    tuple val(model), path("${model}.${shard.baseName}"), emit: shard_dir

    script:
    """
    "${env_prefix}/bin/python" ${embed_py} \\
        --model ${model} \\
        --fasta ${shard} \\
        --residue_sample ${residue_sample} \\
        --device ${params.device} \\
        --precision ${params.precision} \\
        --max_tokens ${params.max_tokens} \\
        --outdir ${model}.${shard.baseName}
    """
}

process MERGE {
    tag "${model}"
    publishDir "${params.outdir}/embeddings", mode: 'copy'

    input:
    tuple val(model), path(shard_dirs)
    path residue_sample
    path merge_py

    output:
    path "${model}", emit: emb_dir

    script:
    """
    "${params.core_prefix}/bin/python" ${merge_py} \\
        --model ${model} \\
        --residue_sample ${residue_sample} \\
        --outdir ${model} \\
        ${shard_dirs}
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
        --pair_sample ${params.pair_sample} \\
        --outdir comparison
    """
}

workflow {
    ['input', 'manifest', 'shards', 'residue_sample', 'core_prefix', 'esm2_prefix', 'esmc_prefix'].each { name ->
        if (!params[name]) {
            error "Missing --${name}. Launch the pipeline through run_server.sh, which sets it."
        }
    }
    fasta = file(params.input, checkIfExists: true)
    manifest = file(params.manifest, checkIfExists: true)
    residue_sample = file(params.residue_sample, checkIfExists: true)
    shards = channel.fromPath("${params.shards}/shard_*.fasta", checkIfExists: true)

    models = channel.of(
        ['esm2_650m', params.esm2_prefix],
        ['esmc_600m', params.esmc_prefix]
    )
    EMBED(models.combine(shards), residue_sample, file("${projectDir}/bin/embed.py", checkIfExists: true))
    MERGE(
        EMBED.out.shard_dir.groupTuple(),
        residue_sample,
        file("${projectDir}/bin/merge_shards.py", checkIfExists: true)
    )
    COMPARE(
        MERGE.out.emb_dir.collect(),
        manifest,
        fasta,
        file("${projectDir}/bin/compare_embeddings.py", checkIfExists: true)
    )
}
