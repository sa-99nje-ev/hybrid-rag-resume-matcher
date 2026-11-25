import unittest
from src.preprocessing import chunks
from src.retrieval import tokenize, bm25, rrf
from src.rag import verify_citation
from src.evaluation import judged_metrics

class Foundations(unittest.TestCase):
    def test_technical_tokens(self):
        self.assertEqual(tokenize('C++ C# .NET Node.js CI/CD'), ['c++', 'c#', '.net', 'node.js', 'ci/cd'])

    def test_chunk_offsets(self):
        text = 'Skills\nC++ and C#\nExperience\nNode.js'
        for part in chunks('x', text, size=12, overlap=2):
            self.assertEqual(part['text'], text[part['start']:part['end']])

    def test_ranking(self):
        self.assertEqual(bm25('C++', {'a': 'C++', 'b': 'Node.js'})[0][0], 'a')
        self.assertEqual(rrf([['a', 'b'], ['b', 'a']])[0][0], 'a')

    def test_citation(self):
        chunk = {'document_id': 'a', 'start': 0, 'end': 3, 'text': 'C++'}
        self.assertTrue(verify_citation({'document_id':'a','chunk_id':'c','quote':'C++'}, {'a':'C++'}, {'c':chunk}))
        self.assertFalse(verify_citation({'document_id':'a','chunk_id':'c','quote':'C#'}, {'a':'C++'}, {'c':chunk}))

    def test_unjudged(self):
        self.assertIsNone(judged_metrics(['a', 'b'], {'a': 1})['metrics'])
        self.assertEqual(judged_metrics(['a'], {'a': 1}, 1)['metrics']['ndcg'], 1.0)

if __name__ == '__main__':
    unittest.main()
