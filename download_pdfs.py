import json
import time
from pathlib import Path
import requests

def download_deduplicated_pdfs():
    # 1. 定义文件夹路径（适配当前项目结构）
    project_root = Path(__file__).resolve().parent
    cfs_root = project_root / "data" / "cf_datasets"
    papers_root = project_root / "data" / "papers"
    
    if not cfs_root.exists() or not papers_root.exists():
        print("❌ 错误：找不到指定的文件夹，请确保脚本与这两个文件夹在同一目录下。")
        return

    print("🔍 [阶段 1] 正在扫描全部反事实文件夹，提取并去重 PDF 链接...")
    unique_pdfs = {}
    paper_meta = {}
    
    # 使用 rglob 递归扫描所有 json 文件
    for json_file in cfs_root.rglob("*.json"):
        if json_file.name == "meta.json":
            continue
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 提取原论文 ID 和对应的 PDF 链接
            paper_id = data.get("o_paper")
            cf_paper = data.get("cf_paper", {})
            meta = cf_paper.get("meta", {})
            venue_dir = meta.get("venue_config", {}).get("id") or meta.get("venue")
            pdf_url = meta.get("pdf_url")
            if not pdf_url:
                links = meta.get("links") or []
                pdf_url = links[1] if len(links) > 1 else None
            
            if paper_id and pdf_url:
                # 字典赋值天然去重：如果多篇 JSON 指向同一个 paper_id，只会保留一个链接
                unique_pdfs[paper_id] = pdf_url
                paper_meta[paper_id] = {
                    "venue_dir": venue_dir,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", []),
                    "pdf_url": pdf_url,
                    "source_cf_file": str(json_file),
                }
                
        except Exception as e:
            print(f"⚠️ 解析 {json_file.name} 时出错: {e}")
            
    print(f"✅ 扫描完毕！共发现 {len(unique_pdfs)} 篇不重复的原始论文链接。")
    print("⏳ [阶段 2] 开始对比本地文件，精准下载缺失的 PDF...\n")

    paper_dirs = {meta_file.parent.name: meta_file.parent for meta_file in papers_root.rglob("meta.json")}

    download_count = 0
    skip_count = 0
    fail_count = 0
    missing_paper_dir_count = 0
    download_delay_seconds = 1.0
    
    # 伪装请求头，防止被 arXiv 等学术网站的防爬虫机制拦截
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    for paper_id, pdf_url in unique_pdfs.items():
        meta_info = paper_meta.get(paper_id, {})
        # 安全性优化：确保证书没问题的话，尽量使用 https
        if pdf_url.startswith("http://"):
            pdf_url = pdf_url.replace("http://", "https://")
            
        paper_dir = paper_dirs.get(paper_id)
        if paper_dir is None:
            venue_dir = meta_info.get("venue_dir") or "unknown_venue"
            paper_dir = papers_root / venue_dir / paper_id
            paper_dir.mkdir(parents=True, exist_ok=True)

            meta_path = paper_dir / "meta.json"
            if not meta_path.exists():
                meta_payload = {
                    "title": meta_info.get("title", ""),
                    "authors": meta_info.get("authors", []),
                    "pdf_url": meta_info.get("pdf_url", pdf_url),
                    "source_cf_file": meta_info.get("source_cf_file", ""),
                }
                meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            paper_dirs[paper_id] = paper_dir
        
        # 定义目标保存路径：papers_v0.1/论文ID/论文ID.pdf
        pdf_path = paper_dir / f"{paper_id}.pdf"
        
        # 断点续传逻辑：如果已经下载过了，直接跳过，不用重新消耗网络
        if pdf_path.exists():
            skip_count += 1
            continue
            
        print(f"⬇️ 正在下载: {paper_id}")
        print(f"   🔗 来源: {pdf_url}")
        
        try:
            response = requests.get(pdf_url, headers=headers, timeout=20)
            if response.status_code == 200:
                pdf_path.write_bytes(response.content)
                download_count += 1
                print(f"   ✅ 成功保存至: {pdf_path}")
            else:
                fail_count += 1
                print(f"   ❌ 下载失败，HTTP 状态码: {response.status_code}")
        except Exception as e:
            fail_count += 1
            print(f"   ❌ 网络请求出错: {e}")
            
        # ⚠️ 适度休眠，避免过快请求触发站点限制
        time.sleep(download_delay_seconds)
        
    print("\n🎉 下载流水线执行完毕！")
    print(f"📊 最终统计: 新增下载 {download_count} 篇 | 本地已有跳过 {skip_count} 篇 | 下载失败 {fail_count} 篇 | 缺少论文目录 {missing_paper_dir_count} 篇")

if __name__ == "__main__":
    download_deduplicated_pdfs()