import tempfile
import unittest
from pathlib import Path

from evals.mcp_comparison.mineru_client import clean_env, contained_markdown
from evals.mcp_comparison.run import markdown_blocks, prepare_corpus
from evals.mcp_comparison.cases import cases


class ComparisonBoundaryTests(unittest.TestCase):
    def test_cloud_child_cannot_inherit_provider_secrets_or_dotenv(self):
        env = clean_env({'PATH':'runtime','DASHSCOPE_API_KEY':'private',
                         'QWEN_API_KEY':'private','MINERU_TOKEN':'private',
                         'PAPERPILOT_API_KEY_FILE':'private-path'})
        self.assertEqual(env['PATH'],'runtime')
        self.assertEqual(env['PYTHON_DOTENV_DISABLED'],'1')
        self.assertFalse(any('private' in str(v) for v in env.values()))

    def test_saved_full_text_wins_over_truncated_inline_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'whole.md'
            path.write_text('full table with all conditions',encoding='utf-8')
            self.assertEqual(contained_markdown({'extract_path':str(path),'content':'preview'},temp),
                             'full table with all conditions')
            with self.assertRaises(ValueError):
                contained_markdown({'extract_path':str(Path(temp).parent/'outside.md')},temp)

    def test_table_cells_are_not_cut_to_fit_chunk_size(self):
        table='<table><tr><td>'+('condition '*500)+'</td><td>82.26</td></tr></table>'
        text='Method paragraph\n\n'+table+'\n\nAnother paragraph'
        blocks=markdown_blocks(text,limit=1000)
        self.assertIn(table,blocks)
        self.assertEqual('\n\n'.join(blocks),text)
        with self.assertRaises(ValueError):
            markdown_blocks('x'*12001)

    def test_original_three_and_unseen_questions_are_preserved(self):
        root=Path(__file__).resolve().parents[1]
        rows=cases(root)
        self.assertEqual(len(rows),11)
        self.assertEqual(sum(bool(r.get('original_demo')) for r in rows),3)
        self.assertEqual(len({r['id'] for r in rows}),11)
        for row in rows[3:]:
            self.assertNotIn('criteria',row['question'])
            self.assertTrue(row['criteria'])

    def test_interrupted_import_is_not_relabelled_as_mcp_success(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'corpus.sqlite3').touch()
            with self.assertRaisesRegex(RuntimeError,'completion receipt'):
                prepare_corpus(None,folder,'mcp')
            (folder/'corpus-build.json').write_text('{}',encoding='utf-8')
            prepare_corpus(None,folder,'mcp')


if __name__ == '__main__':
    unittest.main()
