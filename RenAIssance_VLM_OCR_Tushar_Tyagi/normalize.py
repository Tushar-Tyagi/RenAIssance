"""
Text Normalization Module
=======================================

Provides a pipeline of transformations that normalise both ground-truth
transcriptions and VLM predictions before error-rate comparison.

The pipeline is order-sensitive. The canonical order is:

    1. Accent removal   (preserving ñ/Ñ)
    2. Cedilla → z
    3. Macron / tilde-above expansion
    4. u/v and f/s disambiguation via dictionary look-up
    5. Lower-casing

Nothing is done with line-end hyphens — they are left as-is.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, cast

from spellchecker import SpellChecker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Unicode combining characters that represent a tilde above (̃ ) and a
#: combining macron above (̄ ).  Both are used in historical manuscripts as
#: abbreviation marks (horizontal caps).
_COMBINING_TILDE_ABOVE: Final[str] = "\u0303"
_COMBINING_MACRON_ABOVE: Final[str] = "\u0304"


#: Comprehensive Spanish dictionary provided by pyspellchecker.
#: Used to validate word candidates during normalization.
_SPANISH_DICT: Final[SpellChecker] = SpellChecker(language="es")


# ---------------------------------------------------------------------------
# 1. Accent removal (preserve ñ / Ñ)
# ---------------------------------------------------------------------------


def remove_accents(text: str) -> str:
    """Remove diacritical combining marks, **except** the tilde on ñ / Ñ.

    Strategy
    --------
    1. NFD-decompose the string so that, for example, ``é`` becomes ``e`` +
       *combining acute accent*.
    2. Walk each character:
       - If it is a *combining mark* (Unicode category ``Mn``), drop it
         **unless** the preceding base character was ``n`` or ``N`` and the
         combining mark is the *combining tilde* (U+0303).  In that case
         the pair represents ``ñ`` / ``Ñ`` and must be kept.
    3. NFC-compose the result so any surviving base + combining pairs are
       merged back into precomposed form.

    Args:
        text: Input string (may contain precomposed or decomposed characters).

    Returns:
        String with accents stripped except for ñ / Ñ.
    """
    decomposed: str = unicodedata.normalize("NFD", text)
    result: list[str] = []

    for char in decomposed:
        if unicodedata.category(char) == "Mn":
            # Keep COMBINING TILDE ABOVE only when it follows n/N (→ ñ/Ñ)
            if char == _COMBINING_TILDE_ABOVE and result and result[-1] in ("n", "N"):
                result.append(char)
            # Drop every other combining mark
            continue
        result.append(char)

    return unicodedata.normalize("NFC", "".join(result))


# ---------------------------------------------------------------------------
# 2. Cedilla replacement  ç → z
# ---------------------------------------------------------------------------


def replace_cedilla(text: str) -> str:
    """Replace the archaic grapheme *ç* with modern *z*.

    In early-modern Spanish, ``ç`` (c-cedilla) was commonly used where
    modern orthography uses ``z``.  This function handles both cases.

    Args:
        text: Input string.

    Returns:
        String with ``ç`` → ``z`` and ``Ç`` → ``Z``.
    """
    return text.replace("ç", "z").replace("Ç", "Z")


# ---------------------------------------------------------------------------
# 3. Horizontal-caps (macron / tilde-above) expansion
# ---------------------------------------------------------------------------

#: Regex that matches a *base letter* immediately followed by a combining
#: tilde-above (U+0303) or combining macron (U+0304).
#: Capture groups:
#:   1 — the base letter
#:   2 — the combining mark (not used, but helpful for debugging)
_MACRON_PATTERN_STR: str = rf"([A-Za-z])([{_COMBINING_TILDE_ABOVE}{_COMBINING_MACRON_ABOVE}])"
_MACRON_PATTERN: re.Pattern[str] = cast(re.Pattern[str], re.compile(_MACRON_PATTERN_STR))


def _expand_macron_match(match: re.Match[str]) -> str:
    """Callback for :func:`expand_macrons`.

    Rules
    -----
    * ``n`` or ``N`` + tilde  →  kept as-is (this is ``ñ`` / ``Ñ``)
    * ``q`` + macron/tilde  →  ``que``
    * ``Q`` + macron/tilde  →  ``Que``
    * Any other letter + macron/tilde  →  letter + ``n``

    Args:
        match: A regex match object with group(1) = base letter, group(2) = mark.

    Returns:
        The expanded replacement string.
    """
    base: str = match.group(1)
    mark: str = match.group(2)
    
    # Don't expand the tilde on n / N (that's just ñ / Ñ)
    if base.lower() == "n" and mark == _COMBINING_TILDE_ABOVE:
        return base + mark

    if base == "q":
        return "que"
    if base == "Q":
        return "Que"
    return base + "n"


def expand_macrons(text: str) -> str:
    """Expand letters bearing a horizontal cap (macron or tilde-above).

    In historical manuscripts a macron or tilde placed above a letter
    indicates a following nasal consonant (usually *n*) that has been
    omitted.  The special case ``q̃`` expands to ``que``.

    The input is first NFD-decomposed so that precomposed characters are
    split into base + combining mark; after substitution the string is
    NFC-recomposed.

    Args:
        text: Input string (pre- or decomposed).

    Returns:
        String with macron abbreviations expanded.
    """
    decomposed: str = unicodedata.normalize("NFD", text)
    expanded: str = _MACRON_PATTERN.sub(_expand_macron_match, decomposed)
    return unicodedata.normalize("NFC", expanded)


# ---------------------------------------------------------------------------
# 4. u/v and f/s disambiguation
# ---------------------------------------------------------------------------


def _try_swap(word: str, old: str, new: str) -> str | None:
    """Return ``word`` with *old* → *new* if the result is in the dictionary.

    Only the **first** occurrence is swapped per call.  The caller should
    invoke this function iteratively if multiple swaps might be needed.

    Args:
        word: A single lower-cased word.
        old:  Character to replace (e.g. ``'u'``).
        new:  Replacement character (e.g. ``'v'``).

    Returns:
        The swapped word if it is a valid Spanish word, else ``None``.
    """
    if old not in word:
        return None
    candidate: str = word.replace(old, new, 1)
    # the known() method checks if a word is in the dictionary
    if _SPANISH_DICT.known([candidate]):
        return candidate
    return None


def disambiguate_uv_fs(text: str) -> str:
    """Disambiguate archaic u/v and f/s spellings via dictionary look-up.

    For every word in ``text``, the function strips surrounding punctuation.
    If the base word is ALREADY a valid Spanish word, it is left intact so
    that we do not accidentally corrupt valid text (e.g. converting "fin"
    into "sin" just because both are valid).

    If the word is not in the dictionary, it tries the following swaps
    and keeps the **first** match found:

    * ``u`` → ``v``
    * ``v`` → ``u``
    * ``f`` → ``s``
    * ``s`` → ``f``

    Args:
        text: Input string.

    Returns:
        String with u/v and f/s ambiguities resolved where possible.
    """
    # Regex to pull off non-word characters at the start and end of a token.
    # This ensures we don't accidentally fail dictionary lookups due to commas.
    tokens: list[str] = text.split()
    result: list[str] = []

    for token in tokens:
        # Separate punctuation from the actual word core
        match = re.match(r'^([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)$', token)
        if not match:
            # Token does not contain a clear alphabetic core
            result.append(token)
            continue

        prefix_group, core_group, suffix_group = match.groups()
        prefix = str(prefix_group)
        core = str(core_group)
        suffix = str(suffix_group)
        lower_core = core.lower()

        # 1. Guard against corrupting words that are ALREADY valid
        if _SPANISH_DICT.known([lower_core]):
            result.append(token)
            continue

        # 2. Try each swap; first dictionary hit wins.
        swapped_core: str | None = None
        for old_char, new_char in [("u", "v"), ("v", "u"), ("f", "s"), ("s", "f")]:
            swapped_core = _try_swap(lower_core, old_char, new_char)
            if swapped_core is not None:
                # Preserve original casing pattern as much as possible
                if core[0].isupper():
                    swapped_core = swapped_core.capitalize()
                
                # Reconstruct token with punctuation and append
                result.append(prefix + swapped_core + suffix)
                break
        
        if swapped_core is None:
            # No valid substitution was found, keep original
            result.append(token)

    return " ".join(result)


# ---------------------------------------------------------------------------
# 5. Lower-casing
# ---------------------------------------------------------------------------


def lowercase(text: str) -> str:
    """Convert text to lower case for stable comparison.

    Args:
        text: Input string.

    Returns:
        Lower-cased string.
    """
    return text.lower()


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Run the full paleographic normalisation pipeline.

    The steps are applied in strict order:

    1. Replace ç → z.
    2. Remove accents (preserve ñ).
    3. Expand macron / tilde abbreviations.
    4. Disambiguate u/v and f/s via dictionary.
    5. Lower-case.

    Args:
        text: Raw transcription or model prediction.

    Returns:
        Normalised text suitable for CER / WER comparison.
    """
    text = replace_cedilla(text)
    text = remove_accents(text)
    text = expand_macrons(text)
    text = disambiguate_uv_fs(text)
    text = lowercase(text)
    return text
