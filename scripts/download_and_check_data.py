import os
import sys
import pandas as pd
from wilds import get_dataset

def main():
    root_dir = "data"
    os.makedirs(root_dir, exist_ok=True)
    print("Iniciando verificação do dataset Waterbirds via WILDS...")
    try:
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=root_dir)
        print("Dataset Waterbirds carregado com sucesso!")
        metadata_df = dataset.metadata_df
        print("Colunas do metadata_df:", metadata_df.columns.tolist())
        print(f"Total de amostras: {len(metadata_df)}")
        print("\nDistribuição por split (0=train, 1=val, 2=test):")
        print(metadata_df['split'].value_counts().sort_index())
        
        # y: 0=landbird, 1=waterbird
        # place: 0=land, 1=water
        # group = 2 * y + place
        # 0: landbird on land
        # 1: landbird on water
        # 2: waterbird on land
        # 3: waterbird on water
        if 'y' in metadata_df.columns and 'place' in metadata_df.columns:
            metadata_df['group'] = 2 * metadata_df['y'] + metadata_df['place']
        elif 'y' in metadata_df.columns and 'background' in metadata_df.columns:
            metadata_df['group'] = 2 * metadata_df['y'] + metadata_df['background']
            
        print("\nTabela cruzada (Split x Grupo):")
        split_map = {0: 'Train', 1: 'Val', 2: 'Test'}
        metadata_df['split_name'] = metadata_df['split'].map(split_map)
        ct = pd.crosstab(metadata_df['split_name'], metadata_df['group'], margins=True)
        print(ct)
        
    except Exception as e:
        print(f"Erro ao carregar via WILDS: {e}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    main()
