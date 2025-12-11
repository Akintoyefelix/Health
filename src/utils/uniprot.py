import json
import requests
from pathlib import Path
from typing import Dict, Optional

class UniProtCache:
    """Creating Cached access to UniProt protein database"""

    def __init__(self, cache_dir="./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "uniprot_cache.json"
        self.cache = self._load_cache()

        # Listing some common skin cancer proteins
        self.cancer_proteins = [
            'BRAF', 'TP53', 'NRAS', 'CDKN2A', 'PTEN',
            'KIT', 'NF1', 'MAP2K1', 'TERT', 'ARID2'
        ]

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def preload_cancer_proteins(self):
        """Preloads data for known cancer proteins."""
        for protein in self.cancer_proteins:
            if protein not in self.cache:
                self.fetch_protein_info(protein)

    def fetch_protein_info(self, gene_name: str) -> Optional[Dict]:
        """Fetches protein info from UniProt API or cache."""
        gene_name = gene_name.upper()
        
        if gene_name in self.cache:
            return self.cache[gene_name]

        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene_exact:{gene_name} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,protein_name,comments,sequence",
            "format": "json",
            "size": 1
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                entry = data['results'][0]
                
                # Extract function comment
                function = "Function not available."
                for comment in entry.get('comments', []):
                    if comment['commentType'] == 'FUNCTION':
                        function = comment['texts'][0]['value']
                        break

                info = {
                    "gene": gene_name,
                    "protein_name": entry['proteinDescription']['recommendedName']['fullName']['value'],
                    "accession": entry['primaryAccession'],
                    "function": function,
                    "sequence_length": entry['sequence']['length']
                }
                
                self.cache[gene_name] = info
                self._save_cache()
                return info
            
        except Exception as e:
            print(f"Error fetching UniProt data for {gene_name}: {e}")
        
        return None
