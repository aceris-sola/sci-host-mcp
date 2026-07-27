#!/usr/bin/env python3
"""科学方向探索宿主系统 — 启动入口.

用法:
    python -m sci_host.run

    python -m sci_host.run --iterations 10

    python -m sci_host.run --mode production

    python -m sci_host.run --offline --interval 10 --iterations 20

    python -m sci_host.run --single-step
"""
from __future__ import annotations

import argparse
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="科学方向探索宿主系统 — 7×24 持续运行",
    )
    parser.add_argument(
        "--mode", choices=["quick", "production", "materials", "custom"],
        default="quick",
        help="运行模式: quick(演示) / production(7×24) / materials(材料科学) / custom(自定义)",
    )
    parser.add_argument(
        "--offline", action="store_true", default=True,
        help="使用离线模式（内置语料，不访问网络）",
    )
    parser.add_argument(
        "--online", action="store_true",
        help="使用在线模式（通过 arXiv API 采集真实论文）",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="循环间隔（秒）",
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="最大循环次数（0=无限）",
    )
    parser.add_argument(
        "--single-step", action="store_true",
        help="单步执行模式（执行一轮后退出）",
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="知识库保存路径",
    )
    parser.add_argument(
        "--materials", action="store_true",
        help="使用材料科学模式 (CSP 抽取 + 材料假设模板)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    from sci_host import HostSystem, HostConfig

    
    if args.mode == "production":
        config = HostConfig.production()
        if args.online:
            config.crawler.offline_mode = False
    elif args.mode == "materials" or args.materials:
        config = HostConfig.materials(
            offline=not args.online,
            loop_interval=args.interval,
        )
        if args.iterations != 5:
            config.loop.max_iterations = args.iterations
    elif args.mode == "custom":
        config = HostConfig()
        config.crawler.offline_mode = not args.online
        config.loop.loop_interval = args.interval
        config.loop.max_iterations = args.iterations
        config.loop.save_path = args.save_path
    else:
        
        config = HostConfig.quick(
            offline=not args.online,
            loop_interval=args.interval,
        )
        if args.iterations != 5:
            config.loop.max_iterations = args.iterations

    if args.save_path:
        config.loop.save_path = args.save_path

    
    host = HostSystem(config)

    print("=" * 60)
    print("  科学方向探索宿主系统 (Scientific Direction Host)")
    print("  7×24 持续运行 | 隐性配对 | 试错与重复")
    print("=" * 60)
    print(f"  模式: {args.mode}")
    print(f"  采集: {'离线' if config.crawler.offline_mode else '在线(arXiv)'}")
    print(f"  材料模式: {'是' if config.materials_mode else '否'}")
    print(f"  间隔: {config.loop.loop_interval}s")
    print(f"  迭代: {'无限' if config.loop.max_iterations == 0 else config.loop.max_iterations}")
    print(f"  研究种子: {config.research_seeds}")
    print("=" * 60)
    print()

    if args.single_step:
        
        host.start()
        print("[单步模式] 执行一轮探索循环...")
        result = host.step()
        print()
        print("═══ 探索结果 ═══")
        print(f"  论文采集: {result['papers_crawled']} 篇")
        print(f"  隐性配对: {result['pairs_found']} 对")
        print(f"  假设生成: {result['hypotheses_generated']} 个")
        print(f"  试错通过: {result['trials_passed']} 个")
        print(f"  试错失败: {result['trials_failed']} 个")
        print(f"  耗时: {result['elapsed_ms']:.0f} ms")
        print()
        print(host.status())
        print()

        directions = host.get_top_directions(5)
        if directions:
            print("🎯 发现的研究方向:")
            for i, d in enumerate(directions, 1):
                print(f"  {i}. {d['label']}")
                print(f"     置信度={d['confidence']:.3f} 支持={d['support_count']} "
                      f"类型={d['hypothesis_type']}")
        host.stop()
    else:
        
        host.run()

    
    print()
    print("═══ 最终报告 ═══")
    print(host.status())
    print()

    knowledge = host.get_knowledge_summary()
    if knowledge:
        print(f"知识库: {knowledge.get('validated', 0)} 条已验证, "
              f"{knowledge.get('failed', 0)} 条失败")
        print(f"  平均评分: {knowledge.get('avg_score', 0):.3f}")
        print(f"  平均新颖性: {knowledge.get('avg_novelty', 0):.3f}")
        if knowledge.get("top_keywords"):
            print(f"  高频关键词: {', '.join(kw for kw, _ in knowledge['top_keywords'][:5])}")

    directions = host.get_top_directions(5)
    if directions:
        print()
        print("🎯 最有前景的研究方向:")
        for i, d in enumerate(directions, 1):
            print(f"  {i}. {d['label']}")
            print(f"     置信度={d['confidence']:.3f} 支持证据={d['support_count']}")


if __name__ == "__main__":
    main()
