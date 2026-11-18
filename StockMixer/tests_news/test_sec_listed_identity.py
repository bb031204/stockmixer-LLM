import unittest
from news_fusion.master_data import issuer_owned_ticker,same_filer

class Tests(unittest.TestCase):
    def test_direct_owner_tuple(self):
        self.assertTrue(issuer_owned_ticker('Sierra Bancorp (Nasdaq:BSRR), parent of Bank of the Sierra.','Sierra Bancorp','BSRR','NASDAQ'))
    def test_other_party_symbol_not_owned_by_issuer(self):
        self.assertFalse(issuer_owned_ticker('Nicholas Financial has a loan from Prospect (NASDAQ: PSEC).','Nicholas Financial','PSEC','NASDAQ'))
    def test_legal_suffix_punctuation_is_not_a_different_issuer(self):
        self.assertTrue(issuer_owned_ticker('Example Corporation, (NYSE: EX).','Example Corporation','EX','NYSE'))
        self.assertTrue(issuer_owned_ticker('Example Corp. (NASDAQ: EX).','Example Corp','EX','NASDAQ'))
    def test_index_filer_must_match_entity(self):
        source='<span class="companyName">SIERRA BANCORP (Filer)</span>'
        self.assertTrue(same_filer(source,'Sierra Bancorp'));self.assertFalse(same_filer(source,'Other Bank'))
    def test_legal_suffix_normalization_preserves_distinct_names(self):
        source='<span class="companyName">BRYN MAWR BANK CORP (Filer)</span>'
        self.assertTrue(same_filer(source,'Bryn Mawr Bank Corporation'));self.assertFalse(same_filer(source,'Bryn Mawr Trust'))
    def test_sec_registration_display_suffix_is_not_an_issuer_change(self):
        source='<span class="companyName">COMERICA INC /NEW/ (Filer)</span>'
        self.assertTrue(same_filer(source,'Comerica Incorporated'));self.assertFalse(same_filer(source,'Different Bank'))

if __name__=='__main__':unittest.main()
