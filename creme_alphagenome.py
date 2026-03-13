"""CREME-style CRE experiment framework for AlphaGenome.

Translates the conceptual framework from CREME-NN (Toneyan and Koo, Nature
Genetics 2024; https://www.nature.com/articles/s41588-024-01923-3) into
AlphaGenome API calls: systematic tile-based perturbations for necessity
testing, higher-order interaction discovery, and CRISPRi tiling scans.

Perturbation strategy: each tile's reference sequence is replaced with a
dinucleotide-frequency-preserving shuffle — a standard genomics null model
that disrupts regulatory motifs while keeping local sequence composition
intact. Each shuffle is scored in both forward and reverse-complement
orientations and averaged across multiple replicates.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Dinucleotide shuffle (adapted from deeplift: string-input branch only)
# ---------------------------------------------------------------------------


def dinuc_shuffle(
    seq: str,
    num_shufs: int | None = None,
    rng: np.random.RandomState | None = None,
) -> str | list[str]:
  """Create dinucleotide-frequency-preserving shuffles of a DNA string.

  Args:
    seq: DNA string of length L.
    num_shufs: Number of shuffles to create. If None, returns a single string.
    rng: Random state for reproducibility.

  Returns:
    Shuffled sequence(s) preserving dinucleotide frequencies.
  """
  if rng is None:
    rng = np.random.RandomState()

  arr = np.frombuffer(bytearray(seq, 'utf8'), dtype=np.int8)
  chars, tokens = np.unique(arr, return_inverse=True)

  shuf_next_inds = []
  for t in range(len(chars)):
    mask = tokens[:-1] == t
    inds = np.where(mask)[0]
    shuf_next_inds.append(inds + 1)

  results: list[str] = []
  n = num_shufs if num_shufs else 1
  for _ in range(n):
    for t in range(len(chars)):
      inds = np.arange(len(shuf_next_inds[t]))
      if len(inds) > 1:
        inds[:-1] = rng.permutation(len(inds) - 1)
      shuf_next_inds[t] = shuf_next_inds[t][inds]
    counters = [0] * len(chars)
    ind = 0
    result = np.empty_like(tokens)
    result[0] = tokens[ind]
    for j in range(1, len(tokens)):
      t = tokens[ind]
      ind = shuf_next_inds[t][counters[t]]
      counters[t] += 1
      result[j] = tokens[ind]
    results.append(chars[result].tobytes().decode('ascii'))
  return results if num_shufs else results[0]


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

# Module-level cache for fetched reference sequences.
_SEQ_CACHE: dict[str, str] = {}


def _fetch_reference_sequence(interval: genome.Interval) -> str:
  """Fetch reference sequence from the UCSC REST API (hg38).

  Results are cached in a module-level dict to avoid redundant requests.

  Args:
    interval: Genomic interval to fetch sequence for.

  Returns:
    Uppercase DNA string for the interval.
  """
  key = f'{interval.chromosome}:{interval.start}-{interval.end}'
  if key in _SEQ_CACHE:
    return _SEQ_CACHE[key]

  url = (
      'https://api.genome.ucsc.edu/getData/sequence'
      f'?genome=hg38&chrom={interval.chromosome}'
      f'&start={interval.start}&end={interval.end}'
  )
  resp = requests.get(url, timeout=30)
  resp.raise_for_status()
  seq = resp.json()['dna'].upper()
  _SEQ_CACHE[key] = seq
  return seq


def _reverse_complement(seq: str) -> str:
  """Return the reverse complement of a DNA string."""
  complement = str.maketrans('ACGT', 'TGCA')
  return seq.translate(complement)[::-1]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NecessityResult:
  """Result of a tile-based necessity test.

  Each tile's reference sequence is replaced with a dinucleotide-preserving
  shuffle and the effect on target gene expression is scored via
  GeneMaskLFCScorer, averaged over multiple shuffle replicates in both
  forward and reverse-complement orientations.

  Attributes:
    scores: Tidy DataFrame from variant_scorers.tidy_scores() with all
      per-tile, per-gene, per-biosample scores (averaged over shuffles).
    tile_effects: Per-tile summary DataFrame filtered to the target gene,
      with columns: tile_label, chrom, start, end, raw_score,
      quantile_score, ontology_curie.
    tiles: The list of genome.Interval tiles that were tested.
    target_gene: Gene symbol that was tested.
    context_interval: The model context interval used.
  """

  scores: pd.DataFrame
  tile_effects: pd.DataFrame
  tiles: list[genome.Interval]
  target_gene: str
  context_interval: genome.Interval

  def plot_bar(
      self,
      ontology_curie: str | None = None,
      ax: plt.Axes | None = None,
  ) -> plt.Axes:
    """Horizontal bar chart of tiles ranked by LFC for the target gene.

    Args:
      ontology_curie: Filter to a specific tissue/cell type. If None, uses
        the first ontology_curie in tile_effects.
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    df = self.tile_effects.copy()
    if ontology_curie is not None:
      df = df[df['ontology_curie'] == ontology_curie]
    else:
      curie = df['ontology_curie'].iloc[0]
      df = df[df['ontology_curie'] == curie]
      ontology_curie = curie

    df = df.sort_values('raw_score', key=abs, ascending=True)

    if ax is None:
      _, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(df))))

    colors = ['#2166ac' if v < 0 else '#b2182b' for v in df['raw_score']]
    ax.barh(
        df['tile_label'], df['raw_score'],
        color=colors, edgecolor='none',
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LFC (log2 ALT/REF)')
    ax.set_title(
        f'Necessity test: {self.target_gene} | {ontology_curie}'
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return ax

  def plot_heatmap(self, ax: plt.Axes | None = None) -> plt.Axes:
    """Heatmap of tiles x ontology terms colored by LFC.

    Args:
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    df = self.tile_effects.copy()
    pivot = df.pivot_table(
        index='tile_label',
        columns='ontology_curie',
        values='raw_score',
        aggfunc='first',
    )
    # Order rows by genomic position.
    tile_order = [f'Tile {i+1}' for i in range(len(self.tiles))]
    pivot = pivot.reindex([t for t in tile_order if t in pivot.index])

    if ax is None:
      fig_h = max(3, 0.5 * len(pivot))
      fig_w = max(5, 0.8 * len(pivot.columns) + 2)
      _, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)
    cmap = plt.cm.RdBu_r
    im = ax.imshow(
        pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax,
        aspect='auto', interpolation='nearest',
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        pivot.columns, rotation=45, ha='right', fontsize=8,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f'Necessity: {self.target_gene} (LFC)')
    plt.colorbar(im, ax=ax, label='LFC', shrink=0.8)
    plt.tight_layout()
    return ax

  def plot_genome(
      self,
      transcript_extractor: (
          transcript_utils.TranscriptExtractor | None
      ) = None,
      ontology_curie: str | None = None,
      ax: plt.Axes | None = None,
  ) -> plt.Axes:
    """Genome browser view with colored tile rectangles + transcripts.

    Args:
      transcript_extractor: For drawing gene models. If None, skips.
      ontology_curie: Filter scores to this tissue. Uses first if None.
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    df = self.tile_effects.copy()
    if ontology_curie is not None:
      df = df[df['ontology_curie'] == ontology_curie]
    else:
      curie = df['ontology_curie'].iloc[0]
      df = df[df['ontology_curie'] == curie]
      ontology_curie = curie

    # Determine plot region from tiles.
    region_start = min(t.start for t in self.tiles)
    region_end = max(t.end for t in self.tiles)
    pad = int((region_end - region_start) * 0.1)
    view_interval = genome.Interval(
        self.tiles[0].chromosome,
        region_start - pad,
        region_end + pad,
    )

    # Transcript annotation.
    if transcript_extractor is not None:
      transcripts = transcript_extractor.extract(view_interval)
      plot_components.plot(
          [plot_components.TranscriptAnnotation(transcripts)],
          interval=view_interval,
          title=f'Necessity: {self.target_gene} tiles',
      )
      plt.show()

    # Draw tile rectangles on a separate figure.
    fig_tiles, ax_tiles = plt.subplots(figsize=(14, 2))

    scores_by_tile = {}
    for _, row in df.iterrows():
      scores_by_tile[row['tile_label']] = row['raw_score']

    vmax = (
        max(abs(v) for v in scores_by_tile.values())
        if scores_by_tile
        else 1e-6
    )
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r

    for i, tile in enumerate(self.tiles):
      label = f'Tile {i+1}'
      score = scores_by_tile.get(label, 0.0)
      color = cmap(norm(score))
      rect = mpatches.FancyBboxPatch(
          (tile.start, 0.1), tile.width, 0.8,
          boxstyle='round,pad=0', facecolor=color,
          edgecolor='black', linewidth=0.5,
      )
      ax_tiles.add_patch(rect)
      ax_tiles.text(
          tile.start + tile.width / 2, 0.5, label,
          ha='center', va='center', fontsize=7,
      )

    ax_tiles.set_xlim(view_interval.start, view_interval.end)
    ax_tiles.set_ylim(0, 1)
    ax_tiles.set_xlabel(f'{self.tiles[0].chromosome} position')
    ax_tiles.set_yticks([])
    ax_tiles.set_title(f'Tile necessity scores ({ontology_curie})')
    ax_tiles.spines[['top', 'right', 'left']].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_tiles, label='LFC', shrink=0.6, pad=0.02)
    plt.tight_layout()
    return ax_tiles


@dataclasses.dataclass
class InteractionResult:
  """Result of a greedy multi-tile ablation (higher-order interaction test).

  At each round, the tile with the largest additional effect is removed,
  and the cumulative expression change is tracked.

  Attributes:
    rounds: List of dicts with keys: round (int, 1-indexed), tile_removed
      (str), tile_interval (genome.Interval), delta (float),
      cumulative_effect (float), remaining_scores (pd.DataFrame).
    target_gene: Gene symbol.
    ontology_curie: Tissue used for scoring.
  """

  rounds: list[dict[str, Any]]
  target_gene: str
  ontology_curie: str

  def plot_waterfall(self, ax: plt.Axes | None = None) -> plt.Axes:
    """Cumulative expression change as tiles are removed.

    Step plot showing how expression drops as tiles are ablated,
    labeled with which tile was removed at each step.

    Args:
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    if ax is None:
      _, ax = plt.subplots(figsize=(8, 5))

    rounds_num = [0] + [r['round'] for r in self.rounds]
    cum_effects = [0.0] + [r['cumulative_effect'] for r in self.rounds]

    ax.step(
        rounds_num, cum_effects, where='post', color='#2166ac',
        linewidth=2, marker='o', markersize=6,
    )
    ax.fill_between(
        rounds_num, cum_effects, step='post',
        alpha=0.15, color='#2166ac',
    )

    for r in self.rounds:
      ax.annotate(
          r['tile_removed'],
          xy=(r['round'], r['cumulative_effect']),
          xytext=(5, 10), textcoords='offset points',
          fontsize=8, ha='left',
          arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
      )

    ax.set_xlabel('Tiles removed')
    ax.set_ylabel('Cumulative LFC from WT')
    ax.set_title(
        f'Interaction test: {self.target_gene} | {self.ontology_curie}'
    )
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return ax

  def plot_contribution(self, ax: plt.Axes | None = None) -> plt.Axes:
    """Per-round delta bar chart.

    Args:
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    if ax is None:
      _, ax = plt.subplots(figsize=(8, 5))

    labels = [r['tile_removed'] for r in self.rounds]
    deltas = [r['delta'] for r in self.rounds]
    colors = ['#2166ac' if d < 0 else '#b2182b' for d in deltas]

    ax.bar(labels, deltas, color=colors, edgecolor='none')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Tile removed (in order)')
    ax.set_ylabel('Delta LFC')
    ax.set_title(f'Per-round contribution: {self.target_gene}')
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return ax


@dataclasses.dataclass
class CRISPRiResult:
  """Result of a CRISPRi tiling scan with full track-level outputs.

  Attributes:
    tile_outputs: List of (tile_interval, variant, VariantOutput) tuples
      from predict_variant for each tile perturbation.
    tile_scores: Tidy DataFrame of scalar scores across all tiles.
    tiles: List of genome.Interval tiles tested.
    target_gene: Gene symbol.
    context_interval: Model context interval used.
  """

  tile_outputs: list[tuple[genome.Interval, genome.Variant, Any]]
  tile_scores: pd.DataFrame
  tiles: list[genome.Interval]
  target_gene: str
  context_interval: genome.Interval

  def plot_tracks(
      self,
      tile_idx: int,
      modality: str = 'rna_seq',
      transcript_extractor: (
          transcript_utils.TranscriptExtractor | None
      ) = None,
      zoom_width: int = 2**15,
  ) -> None:
    """Overlaid WT/CRISPRi tracks for one tile perturbation.

    Args:
      tile_idx: Index into tiles/tile_outputs to plot.
      modality: Which modality to display (e.g. 'rna_seq', 'dnase').
      transcript_extractor: For gene annotation track. Optional.
      zoom_width: Width to zoom the view to.
    """
    tile, variant, voutput = self.tile_outputs[tile_idx]

    ref_tdata = getattr(voutput.reference, modality)
    alt_tdata = getattr(voutput.alternate, modality)

    components = []
    if transcript_extractor is not None:
      transcripts = transcript_extractor.extract(self.context_interval)
      components.append(
          plot_components.TranscriptAnnotation(transcripts)
      )

    components.append(
        plot_components.OverlaidTracks(
            tdata={'WT': ref_tdata, 'CRISPRi': alt_tdata},
            colors={'WT': 'dimgrey', 'CRISPRi': 'red'},
        )
    )

    view_iv = (
        ref_tdata.interval.resize(zoom_width)
        if zoom_width
        else ref_tdata.interval
    )

    plot_components.plot(
        components,
        interval=view_iv,
        annotations=[plot_components.VariantAnnotation([variant])],
        title=f'CRISPRi Tile {tile_idx+1}: {tile} | {modality}',
    )
    plt.show()

  def plot_summary(
      self,
      ontology_curie: str | None = None,
      ax: plt.Axes | None = None,
  ) -> plt.Axes:
    """Summary bar of expression change per tile.

    Args:
      ontology_curie: Filter to specific tissue. Uses first if None.
      ax: Matplotlib axes to plot on. Created if None.

    Returns:
      The matplotlib Axes object.
    """
    df = self.tile_scores.copy()
    df = df[df['gene_name'] == self.target_gene]

    if ontology_curie is not None:
      df = df[df['ontology_curie'] == ontology_curie]
    elif 'ontology_curie' in df.columns and len(df) > 0:
      curie = df['ontology_curie'].iloc[0]
      df = df[df['ontology_curie'] == curie]
      ontology_curie = curie

    # Use the tile_label column added during crispri_scan.
    tile_labels = []
    scores = []
    for i in range(len(self.tiles)):
      label = f'Tile {i+1}'
      tile_df = df[df['tile_label'] == label]
      if len(tile_df) > 0:
        scores.append(tile_df['raw_score'].iloc[0])
      else:
        scores.append(0.0)
      tile_labels.append(label)

    if ax is None:
      _, ax = plt.subplots(
          figsize=(8, max(3, 0.4 * len(tile_labels)))
      )

    colors = ['#2166ac' if s < 0 else '#b2182b' for s in scores]
    y_pos = range(len(tile_labels))
    ax.barh(y_pos, scores, color=colors, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tile_labels)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LFC (log2 ALT/REF)')
    ax.set_title(
        f'CRISPRi scan: {self.target_gene}'
        + (f' | {ontology_curie}' if ontology_curie else '')
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Variant helpers
# ---------------------------------------------------------------------------


def _make_shuffle_variant(
    tile: genome.Interval,
    ref_sequence: str,
    shuffled_sequence: str,
    name: str | None = None,
) -> genome.Variant:
  """Create a substitution variant replacing ref with a shuffled sequence."""
  return genome.Variant(
      chromosome=tile.chromosome,
      position=tile.start + 1,  # 1-based
      reference_bases=ref_sequence,
      alternate_bases=shuffled_sequence,
      name=name or f'shuf_{tile.chromosome}:{tile.start}-{tile.end}',
  )


def _make_deletion_variant(
    tile: genome.Interval,
    name: str | None = None,
) -> genome.Variant:
  """Create a deletion variant for a tile (ref='N'*width, alt='N').

  Kept for backward-compatible annotation in plots.
  """
  return genome.Variant(
      chromosome=tile.chromosome,
      position=tile.start + 1,  # 1-based
      reference_bases='N' * tile.width,
      alternate_bases='N',
      name=name or f'del_{tile.chromosome}:{tile.start}-{tile.end}',
  )


def _average_shuffle_scores(
    dfs: list[pd.DataFrame],
) -> pd.DataFrame:
  """Average raw_score and quantile_score across shuffle replicates.

  Args:
    dfs: DataFrames from individual shuffle replicates, each with columns
      including raw_score, quantile_score, and grouping columns.

  Returns:
    Single DataFrame with scores averaged over replicates.
  """
  combined = pd.concat(dfs, ignore_index=True)
  # Drop per-replicate columns before grouping.
  drop_cols = {'raw_score', 'quantile_score', 'variant_id', 'scored_interval'}
  # Also drop any columns with unhashable types (e.g. Variant objects).
  for c in combined.columns:
    if c in drop_cols:
      continue
    try:
      combined[c].unique()
    except TypeError:
      drop_cols.add(c)
  group_cols = [c for c in combined.columns if c not in drop_cols]
  averaged = (
      combined
      .groupby(group_cols, as_index=False, sort=False, dropna=False)
      .agg(
          raw_score=('raw_score', 'mean'),
          quantile_score=('quantile_score', 'mean'),
      )
  )
  return averaged


# ---------------------------------------------------------------------------
# Cell type search
# ---------------------------------------------------------------------------


def search_cell_types(
    model: dna_client.DnaClient,
    query: str,
    _cache: dict[str, pd.DataFrame] = {},
) -> pd.DataFrame:
  """Search for available cell types / biosamples by free-text query.

  Runs a lightweight dummy score_variant() call to discover all available
  biosamples, caches the result, then filters by case-insensitive substring
  match across ontology_curie, biosample_name, biosample_type, and
  biosample_life_stage.

  Args:
    model: AlphaGenome DNA model client.
    query: Free-text search string (e.g. 'CD34', 'liver', 'K562').

  Returns:
    DataFrame of matching biosamples, sorted by biosample_name.
  """
  if 'all_biosamples' not in _cache:
    # Lightweight call to discover biosamples.
    dummy_interval = genome.Interval(
        'chr1', 0, dna_client.SEQUENCE_LENGTH_16KB,
    )
    dummy_variant = genome.Variant(
        chromosome='chr1',
        position=100,
        reference_bases='A',
        alternate_bases='T',
        name='dummy',
    )
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]
    result = model.score_variant(
        interval=dummy_interval,
        variant=dummy_variant,
        variant_scorers=scorers,
    )
    df = variant_scorers.tidy_scores([result], match_gene_strand=True)
    # Extract unique biosample info.
    biosample_cols = [
        c for c in df.columns
        if c.startswith(('ontology_curie', 'biosample'))
    ]
    if not biosample_cols:
      biosample_cols = ['ontology_curie']
    biosamples = df[biosample_cols].drop_duplicates().reset_index(drop=True)
    _cache['all_biosamples'] = biosamples

  biosamples = _cache['all_biosamples']
  q = query.lower()
  mask = biosamples.apply(
      lambda row: any(q in str(v).lower() for v in row), axis=1,
  )
  return biosamples[mask].sort_values(
      biosamples.columns[0],
  ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------


class CREExperiment:
  """CREME-style cis-regulatory element experiment runner for AlphaGenome.

  Provides systematic tile-based perturbation tests using dinucleotide
  shuffles as the null model:
  - Necessity test: which tiles are required for target gene expression?
  - Interaction test: greedy ablation to find higher-order CRE interactions.
  - CRISPRi scan: fine-resolution tiling with full track-level outputs.

  Attributes:
    model: AlphaGenome DNA model client.
    gene_symbol: Target gene symbol.
    gtf: Gene annotation DataFrame.
    transcript_extractor: TranscriptExtractor for visualization.
    ontology_terms: List of ontology CURIEs.
    seq_length: Sequence length for model context window.
    gene_interval: Resolved gene interval.
    context_interval: Model context interval (gene resized to seq_length).
  """

  def __init__(
      self,
      model: dna_client.DnaClient,
      gene_symbol: str,
      gtf: pd.DataFrame,
      transcript_extractor: transcript_utils.TranscriptExtractor,
      ontology_terms: list[str],
      seq_length: int = dna_client.SEQUENCE_LENGTH_1MB,
  ):
    self.model = model
    self.gene_symbol = gene_symbol
    self.gtf = gtf
    self.transcript_extractor = transcript_extractor
    self.ontology_terms = ontology_terms
    self.seq_length = seq_length

    # Resolve gene interval.
    self.gene_interval = gene_annotation.get_gene_interval(
        gtf, gene_symbol=gene_symbol,
    )
    self.context_interval = self.gene_interval.resize(seq_length)

  @staticmethod
  def tile_region(
      region: genome.Interval,
      tile_width: int = 5000,
      step: int | None = None,
  ) -> list[genome.Interval]:
    """Tile a genomic region into evenly spaced intervals.

    Args:
      region: The genomic region to tile.
      tile_width: Width of each tile in bp.
      step: Step size between tile starts. Defaults to tile_width
        (non-overlapping).

    Returns:
      List of genome.Interval tiles covering the region.
    """
    if step is None:
      step = tile_width

    tiles = []
    pos = region.start
    while pos < region.end:
      end = min(pos + tile_width, region.end)
      tiles.append(genome.Interval(region.chromosome, pos, end))
      pos += step
    return tiles

  def necessity_test(
      self,
      tiles: list[genome.Interval],
      scorers: list | None = None,
      target_gene: str | None = None,
      n_shuffles: int = 10,
  ) -> NecessityResult:
    """Run a tile-based necessity test with dinucleotide shuffle perturbations.

    For each tile:
    1. Fetches the reference sequence from UCSC.
    2. Generates n_shuffles dinucleotide-preserving shuffles.
    3. Scores each shuffle in both forward and reverse-complement
       orientations (2 * n_shuffles variants per tile).
    4. Averages scores across all replicates.

    Args:
      tiles: List of genomic intervals to perturb one at a time.
      scorers: Variant scorers to use. Defaults to recommended RNA_SEQ
        scorer (GeneMaskLFCScorer).
      target_gene: Gene to focus on. Defaults to self.gene_symbol.
      n_shuffles: Number of dinucleotide shuffle replicates per tile.
        Each replicate is scored in both forward and RC orientations,
        so total API calls per tile = 2 * n_shuffles.

    Returns:
      NecessityResult with averaged scores, tile_effects, and
      visualization methods.
    """
    target_gene = target_gene or self.gene_symbol
    if scorers is None:
      scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]

    rng = np.random.RandomState(42)
    per_tile_dfs = []

    for i, tile in enumerate(tqdm(tiles, desc='Necessity test')):
      ref_seq = _fetch_reference_sequence(tile)
      shuffled_seqs = dinuc_shuffle(
          ref_seq, num_shufs=n_shuffles, rng=rng,
      )

      replicate_dfs = []
      for s, shuf_seq in enumerate(shuffled_seqs):
        # Forward orientation.
        fwd_variant = _make_shuffle_variant(
            tile, ref_seq, shuf_seq,
            name=f'Tile_{i+1}_shuf{s+1}_fwd',
        )
        fwd_result = self.model.score_variant(
            interval=self.context_interval,
            variant=fwd_variant,
            variant_scorers=scorers,
        )
        fwd_df = variant_scorers.tidy_scores(
            [fwd_result], match_gene_strand=True,
        )
        replicate_dfs.append(fwd_df)

        # Reverse complement orientation.
        rc_variant = _make_shuffle_variant(
            tile, ref_seq, _reverse_complement(shuf_seq),
            name=f'Tile_{i+1}_shuf{s+1}_rc',
        )
        rc_result = self.model.score_variant(
            interval=self.context_interval,
            variant=rc_variant,
            variant_scorers=scorers,
        )
        rc_df = variant_scorers.tidy_scores(
            [rc_result], match_gene_strand=True,
        )
        replicate_dfs.append(rc_df)

      # Average across all 2*n_shuffles replicates for this tile.
      avg_df = _average_shuffle_scores(replicate_dfs)
      avg_df['tile_idx'] = i
      avg_df['tile_label'] = f'Tile {i+1}'
      avg_df['chrom'] = tile.chromosome
      avg_df['tile_start'] = tile.start
      avg_df['tile_end'] = tile.end
      per_tile_dfs.append(avg_df)

    scores_df = pd.concat(per_tile_dfs, ignore_index=True)

    # Build tile_effects: filter to target gene.
    gene_scores = scores_df[
        scores_df['gene_name'] == target_gene
    ].copy()

    return NecessityResult(
        scores=scores_df,
        tile_effects=gene_scores,
        tiles=tiles,
        target_gene=target_gene,
        context_interval=self.context_interval,
    )

  def interaction_test(
      self,
      tiles: list[genome.Interval],
      scorers: list | None = None,
      target_gene: str | None = None,
      num_rounds: int | None = None,
      ontology_curie: str | None = None,
      n_shuffles: int = 10,
  ) -> InteractionResult:
    """Greedy multi-tile ablation test (higher-order interaction).

    At each round:
    1. Score all remaining tiles via necessity test (with dinucleotide
       shuffle perturbations).
    2. Pick the tile with the largest |effect| on the target gene.
    3. Record its effect and remove it from the candidate set.
    4. Accumulate the total expression change.

    This approximates CREME's greedy interaction test under a linear
    additivity assumption (each tile's effect is scored independently).

    Args:
      tiles: List of genomic intervals to test.
      scorers: Variant scorers. Defaults to recommended RNA_SEQ.
      target_gene: Gene to focus on. Defaults to self.gene_symbol.
      num_rounds: Max ablation rounds. Defaults to len(tiles).
      ontology_curie: Tissue to use for picking the worst tile.
        Defaults to first available.
      n_shuffles: Number of dinucleotide shuffle replicates per tile.

    Returns:
      InteractionResult with per-round ablation data and plots.
    """
    target_gene = target_gene or self.gene_symbol
    if scorers is None:
      scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]
    if num_rounds is None:
      num_rounds = len(tiles)
    num_rounds = min(num_rounds, len(tiles))

    remaining = list(range(len(tiles)))  # indices into tiles
    rounds = []
    cumulative_effect = 0.0

    for round_num in range(1, num_rounds + 1):
      # Score remaining tiles.
      round_tiles = [tiles[idx] for idx in remaining]
      nec = self.necessity_test(
          round_tiles,
          scorers=scorers,
          target_gene=target_gene,
          n_shuffles=n_shuffles,
      )

      # Get per-tile effects for the target gene.
      effects = nec.tile_effects
      if ontology_curie is not None:
        effects = effects[effects['ontology_curie'] == ontology_curie]
      elif len(effects) > 0:
        ontology_curie = effects['ontology_curie'].iloc[0]

      if len(effects) == 0:
        break

      # Pick tile with largest absolute effect.
      worst_idx = effects['raw_score'].abs().idxmax()
      worst_row = effects.loc[worst_idx]
      worst_tile_label = worst_row['tile_label']
      worst_delta = worst_row['raw_score']

      # Map back to remaining index.
      tile_num = int(worst_tile_label.split()[-1]) - 1
      original_idx = remaining[tile_num]

      cumulative_effect += worst_delta

      rounds.append({
          'round': round_num,
          'tile_removed': f'Tile {original_idx + 1}',
          'tile_interval': tiles[original_idx],
          'delta': float(worst_delta),
          'cumulative_effect': float(cumulative_effect),
          'remaining_scores': effects,
      })

      remaining.pop(tile_num)
      if not remaining:
        break

    return InteractionResult(
        rounds=rounds,
        target_gene=target_gene,
        ontology_curie=ontology_curie or '',
    )

  def crispri_scan(
      self,
      tiles: list[genome.Interval],
      outputs: list[dna_client.OutputType] | None = None,
      target_gene: str | None = None,
      n_shuffles: int = 10,
  ) -> CRISPRiResult:
    """CRISPRi tiling scan with track-level REF vs ALT comparisons.

    For each tile:
    1. Fetches the reference sequence and generates dinucleotide shuffles.
    2. Scalar scoring: averages score_variant across 2*n_shuffles
       replicates (forward + RC for each shuffle).
    3. Track-level prediction: uses the first forward shuffle only
       (tracks are for visual inspection).

    Args:
      tiles: List of genomic intervals to scan.
      outputs: Requested output modalities. Defaults to RNA_SEQ.
      target_gene: Gene of interest. Defaults to self.gene_symbol.
      n_shuffles: Number of dinucleotide shuffle replicates per tile.

    Returns:
      CRISPRiResult with track outputs and summary scores.
    """
    target_gene = target_gene or self.gene_symbol
    if outputs is None:
      outputs = [dna_client.OutputType.RNA_SEQ]

    rng = np.random.RandomState(42)
    tile_outputs = []
    per_tile_dfs = []
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']]

    for i, tile in enumerate(tqdm(tiles, desc='CRISPRi scan')):
      ref_seq = _fetch_reference_sequence(tile)
      shuffled_seqs = dinuc_shuffle(
          ref_seq, num_shufs=n_shuffles, rng=rng,
      )

      # First forward shuffle is used for track-level prediction.
      first_fwd_variant = _make_shuffle_variant(
          tile, ref_seq, shuffled_seqs[0],
          name=f'CRISPRi_Tile_{i+1}_shuf1_fwd',
      )

      # Track-level prediction (first shuffle only).
      voutput = self.model.predict_variant(
          interval=self.context_interval,
          variant=first_fwd_variant,
          requested_outputs=outputs,
          ontology_terms=self.ontology_terms,
      )
      tile_outputs.append((tile, first_fwd_variant, voutput))

      # Scalar scoring: average across all shuffles x 2 orientations.
      replicate_dfs = []
      for s, shuf_seq in enumerate(shuffled_seqs):
        fwd_variant = _make_shuffle_variant(
            tile, ref_seq, shuf_seq,
            name=f'CRISPRi_Tile_{i+1}_shuf{s+1}_fwd',
        )
        fwd_result = self.model.score_variant(
            interval=self.context_interval,
            variant=fwd_variant,
            variant_scorers=scorers,
        )
        fwd_df = variant_scorers.tidy_scores(
            [fwd_result], match_gene_strand=True,
        )
        replicate_dfs.append(fwd_df)

        rc_variant = _make_shuffle_variant(
            tile, ref_seq, _reverse_complement(shuf_seq),
            name=f'CRISPRi_Tile_{i+1}_shuf{s+1}_rc',
        )
        rc_result = self.model.score_variant(
            interval=self.context_interval,
            variant=rc_variant,
            variant_scorers=scorers,
        )
        rc_df = variant_scorers.tidy_scores(
            [rc_result], match_gene_strand=True,
        )
        replicate_dfs.append(rc_df)

      avg_df = _average_shuffle_scores(replicate_dfs)
      avg_df['tile_idx'] = i
      avg_df['tile_label'] = f'Tile {i+1}'
      per_tile_dfs.append(avg_df)

    tile_scores = pd.concat(per_tile_dfs, ignore_index=True)

    return CRISPRiResult(
        tile_outputs=tile_outputs,
        tile_scores=tile_scores,
        tiles=tiles,
        target_gene=target_gene,
        context_interval=self.context_interval,
    )
