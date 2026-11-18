import copy,json,tempfile,unittest
from pathlib import Path
from news_fusion.industry_scope_review import apply_scope_reviews
from news_fusion.io import write_json,write_records,file_manifest,digest
from news_fusion.routing import article_pin


class IndustryScopeTests(unittest.TestCase):
    def setUp(self):
        self.article={'source_id':'a','title':'','summary':'One company expands its credit facility.',
            'published_at':'2013-01-30','input_text_hash':'a'*64,'market':'US',
            'entities':[{'name':'Company'}],'industry_impacts':[{'industry_name':'Services','evidence':'One company expands its credit facility.'}]}
        self.review={'source_id':'a','article_pin':article_pin(self.article),'decision':'exclude',
            'review_method':'assistant_secondary_review','reviewed_by':'Codex assistant','reviewed_at':'2026-11-17T00:00:00Z',
            'rationale':'Only a company financing event; no industry-wide evidence.'}
    def package(self,root,row,confirmation=None):
        write_records(root/'industry_reviews.jsonl',[row]);write_json(root/'user_confirmation.json',confirmation or {})
        write_json(root/'manifest.json',{'kind':'industry_scope_review','snapshot_manifest_sha256':'b'*64,'files':file_manifest(root)})
    def test_unreviewed_industry_can_be_pending_without_losing_company(self):
        rows,stats=apply_scope_reviews([self.article],required=True)
        self.assertEqual(rows[0]['industry_impacts'],[]);self.assertEqual(rows[0]['entities'],self.article['entities'])
        self.assertEqual(stats['pending_impacts'],1);self.assertEqual(self.article['industry_impacts'][0]['industry_name'],'Services')
    def test_optional_policy_preserves_existing_behavior(self):
        rows,stats=apply_scope_reviews([self.article]);self.assertEqual(rows[0],self.article);self.assertEqual(stats['retained_impacts'],1)
    def test_assistant_restriction_is_not_human_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.package(root,self.review);rows,stats=apply_scope_reviews([self.article],root,True,'b'*64)
            self.assertEqual(rows[0]['industry_impacts'],[]);self.assertEqual(stats['human_confirmed_samples'],0)
            self.assertEqual(stats['assistant_reviewed_samples'],1)
    def test_assistant_cannot_approve_sector_propagation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.package(root,dict(self.review,decision='include'))
            with self.assertRaisesRegex(ValueError,'cannot approve'):apply_scope_reviews([self.article],root)
    def test_human_label_requires_actual_matching_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row=dict(self.review,review_method='human_confirmation',reviewed_by='current_session_user');self.package(root,row)
            with self.assertRaisesRegex(ValueError,'actual user'):apply_scope_reviews([self.article],root)
            confirmation={'response':'认可','reviewed_by':'current_session_user','samples':{'a':{'article_pin':row['article_pin'],'decision':'exclude'}}}
            self.package(root,row,confirmation);_,stats=apply_scope_reviews([self.article],root)
            self.assertEqual(stats['human_confirmed_samples'],1)
            confirmation['response']='是公司';self.package(root,row,confirmation)
            _,stats=apply_scope_reviews([self.article],root);self.assertEqual(stats['human_confirmed_samples'],1)
    def test_other_snapshot_or_article_version_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.package(root,self.review)
            with self.assertRaisesRegex(ValueError,'snapshot'):apply_scope_reviews([self.article],root,snapshot_sha='c'*64)
            changed=dict(self.article,input_text_hash='d'*64)
            with self.assertRaisesRegex(ValueError,'pin mismatch'):apply_scope_reviews([changed],root)
    def test_positive_human_approval_cannot_expand_to_changed_model_proposals(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);row=dict(self.review,decision='include',review_method='human_confirmation',reviewed_by='current_session_user',
                evidence_quote=self.article['summary'],approved_impacts_hash=digest(self.article['industry_impacts']))
            confirmation={'response':'认可','reviewed_by':'current_session_user','samples':{'a':{'article_pin':row['article_pin'],'decision':'include'}}}
            self.package(root,row,confirmation);_,stats=apply_scope_reviews([self.article],root)
            self.assertEqual(stats['retained_impacts'],1)
            accepted,_=apply_scope_reviews([self.article],root,preserve_approval_time=True)
            self.assertEqual(accepted[0]['industry_scope_review']['reviewed_at'],row['reviewed_at'])
            changed=copy.deepcopy(self.article);changed['industry_impacts'][0]['industry_name']='Another industry'
            with self.assertRaisesRegex(ValueError,'specific model proposals'):apply_scope_reviews([changed],root)


if __name__=='__main__':unittest.main()
