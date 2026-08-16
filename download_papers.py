import os
import urllib.request
import urllib.error
import ssl

PAPERS = [
    {
        "filename": "01_stojanovic2026_switching_successor_measures.pdf",
        "url": "https://arxiv.org/pdf/2605.13207.pdf",
        "title": "Switching Successor Measures for Hierarchical Zero-shot Reinforcement Learning (Stojanovic & Proutiere, 2026)"
    },
    {
        "filename": "02_touati2021_learning_one_representation.pdf",
        "url": "https://arxiv.org/pdf/2103.07945.pdf",
        "title": "Learning One Representation to Optimize All Rewards (Touati & Ollivier, 2021)"
    },
    {
        "filename": "03_touati2023_does_zero_shot_rl_exist.pdf",
        "url": "https://arxiv.org/pdf/2209.14935.pdf",
        "title": "Does Zero-Shot Reinforcement Learning Exist? (Touati, Rapin & Ollivier, 2023)"
    },
    {
        "filename": "04_barreto2017_successor_features.pdf",
        "url": "https://arxiv.org/pdf/1606.05312.pdf",
        "title": "Successor Features for Transfer in Reinforcement Learning (Barreto et al., 2017)"
    },
    {
        "filename": "05_park2024_ogbench.pdf",
        "url": "https://arxiv.org/pdf/2410.20092.pdf",
        "title": "OGBench: Benchmarking Offline Goal-Conditioned RL (Park et al., 2024)"
    },
    {
        "filename": "06_borsa2018_universal_successor_features.pdf",
        "url": "https://arxiv.org/pdf/1812.07626.pdf",
        "title": "Universal Successor Features Approximators (Borsa et al., 2018)"
    },
    {
        "filename": "07_blier2021_learning_successor_states_and_goals.pdf",
        "url": "https://arxiv.org/pdf/2101.04524.pdf",
        "title": "Learning Successor States and Goals for Transfer and Generalization (Blier et al., 2021)"
    },
    {
        "filename": "08_eysenbach2022_contrastive_rl.pdf",
        "url": "https://arxiv.org/pdf/2206.07568.pdf",
        "title": "Contrastive Learning as Goal-Conditioned Reinforcement Learning (Eysenbach et al., 2022)"
    }
]

def download_all():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")
    os.makedirs(target_dir, exist_ok=True)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    print(f"Downloading {len(PAPERS)} research papers to {target_dir}...\n")
    
    success_count = 0
    for idx, paper in enumerate(PAPERS, start=1):
        target_path = os.path.join(target_dir, paper["filename"])
        print(f"[{idx}/{len(PAPERS)}] {paper['title']}")
        print(f"       URL: {paper['url']}")
        
        try:
            req = urllib.request.Request(paper["url"], headers=headers)
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                content = response.read()
                
            if content.startswith(b"%PDF"):
                with open(target_path, "wb") as f:
                    f.write(content)
                size_mb = len(content) / (1024 * 1024)
                print(f"       -> Saved: {paper['filename']} ({size_mb:.2f} MB) [VALID PDF]")
                success_count += 1
            else:
                print(f"       -> WARNING: Downloaded content is not a PDF (got {len(content)} bytes).")
        except Exception as e:
            print(f"       -> ERROR: {e}")
        print()

    print("=" * 60)
    print(f"Download complete: {success_count}/{len(PAPERS)} papers downloaded successfully.")
    print("=" * 60)

if __name__ == "__main__":
    download_all()
