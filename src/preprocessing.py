"""Section-aware processing with offsets into unchanged source documents."""
import hashlib
import re

HEADINGS = re.compile(r"(?im)^\s*(overview|financial statements|balance sheets|income statements|cash flows|risk factors|notes|revenue|liabilities)\s*:?[ \t]*$")


def stable_id(kind, source_id):
    return hashlib.sha256((kind + '\0' + str(source_id)).encode('utf-8')).hexdigest()


def sections(text):
    matches = list(HEADINGS.finditer(text))
    if not matches:
        return [('document', 0, len(text))]
    result = []
    if matches[0].start() > 0:
        result.append(('document', 0, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append((match.group(1).lower(), match.start(), end))
    return result


def chunks(doc_id, text, size=1000, overlap=150):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('Require size > overlap >= 0')
    for section, start, end in sections(text):
        pos = start
        while pos < end:
            stop = min(pos + size, end)
            chunk_id = stable_id('chunk', f'{doc_id}:{pos}:{stop}')
            yield {'document_id': doc_id, 'chunk_id': chunk_id, 'section': section,
                   'start': pos, 'end': stop, 'text': text[pos:stop],
                   'section_start': start, 'section_end': end}
            if stop == end:
                break
            pos = stop - overlap


def normalize(text):
    return ' '.join(text.split()).casefold()
