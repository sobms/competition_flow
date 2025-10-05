#!/usr/bin/env python3
"""
Простой тест prepare_dataset без Hydra.
"""

import sys
import logging
import time
from pathlib import Path
from omegaconf import OmegaConf

# Добавляем путь к модулям проекта
sys.path.append(str(Path(__file__).parent))

from ml.models import SASRecAdapter

# Включаем INFO-логирование для отображения прогресса prepare_dataset
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logging.getLogger("ml.models.recbole_base").setLevel(logging.INFO)


def main():
    """Простой запуск prepare_dataset с минимальной конфигурацией."""
    
    # Создаем минимальную конфигурацию
    cfg_dict = {
        "seed": 42,
        "artifacts_dir": "artifacts",
        "model": {
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id", 
            "LABEL_FIELD": "target",
            "TIME_FIELD": "date",
            "inter_features_dir": "data/processed_features/inter_features",
            "items_dir": "data/processed_features/items_features", 
            "users_dir": "data/processed_features/users_features",
            "base_cols": [
                {"name": "user_id", "type": "token"},
                {"name": "item_id", "type": "token"},
                {"name": "target", "type": "float"},
                {"name": "date", "type": "float"}
            ],
            "inter_feature_cols": [
                {"name": "cnt_view_description_sum_30d_past", "type": "int"},
                {"name": "cnt_page_view_sum_30d_past", "type": "int"},
                {"name": "cnt_favorite_sum_30d_past", "type": "int"},
                {"name": "cnt_unfavorite_sum_30d_past", "type": "int"},
                {"name": "cnt_to_cart_sum_30d_past", "type": "int"},
                {"name": "cnt_remove_sum_30d_past", "type": "int"},
                {"name": "cnt_review_view_sum_30d_past", "type": "int"}
            ],
            "item_feature_cols": [
                {"name": "fclip_embed", "type": "float_seq"},
                # {"name": "catalogid", "type": "token"},
                # {"name": "variant_id", "type": "token"},
                # {"name": "model_id", "type": "token"}
            ],
            "auxiliary_feature_cols": [
                {"name": "item_id_list", "type": "float_seq"},
                {"name": "item_length", "type": "float"}
            ]
        }
    }
    
    cfg = OmegaConf.create(cfg_dict)
    
    print("Инициализация RecBoleBaseAdapter...")
    
    # Параметры адаптера для SASRec (RecBole конфиг ключи передаем плоско)
    adapter_params = {
        "model": "SASRec",
        "batch_size": 512,           # для нашего DataLoader
        "use_gpu": False,            # тест без GPU
        "infer_batch_size": 512,
        # SASRec/RecBole гиперпараметры
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_prob": 0.2,
        "attn_dropout_prob": 0.2,
        "hidden_dropout_prob": 0.2,
        "max_seq_length": 50,        # длина истории, также используется нашим датасетом
        "learning_rate": 0.001,
        "epochs": 1,                 # быстрее для теста
        "train_batch_size": 1024,
        "eval_batch_size": 1024,
        # отключаем негативное семплирование для CE в SASRec
        "train_neg_sample_args": None,
        "eval_neg_sample_args": None,
    }
    
    adapter = SASRecAdapter(**adapter_params)
    
    print("Запуск prepare_dataset...")
    try:
        train_data, valid_data = adapter.prepare_dataset(cfg)
        
        print("\n✅ prepare_dataset выполнен успешно!")
        print(f"Train data keys: {list(train_data.keys())}")
        print(f"Valid data keys: {list(valid_data.keys())}")
        
        print(f"\nСтатистика:")
        print(f"- Пользователей: {len(adapter.user_id_to_idx)}")
        print(f"- Товаров: {len(adapter.item_id_to_idx)}")
        
        if valid_data.get("users"):
            print(f"- Валидационных пользователей: {len(valid_data['users'])}")
        
        if valid_data.get("ground_truth"):
            print(f"- Ground truth пар: {len(valid_data['ground_truth'])}")
            
        # Проверим один батч
        train_loader = train_data.get("train_loader")
        if train_loader is not None:
            print(f"\n📊 Информация о train_loader:")
            print(f"- Batch size: {train_loader.batch_size}")
            
            # Измерим время первых трех батчей
            print("[TEST] Timing first 3 batches...")
            it = iter(train_loader)
            for i in range(3):
                try:
                    t0 = time.time()
                    b = next(it)
                    dt = time.time() - t0
                    # Определим размер батча
                    bs = 0
                    try:
                        if hasattr(b, 'interaction'):
                            keys_b = list(b.interaction.keys())
                            for k in keys_b:
                                v = b.interaction[k]
                                if hasattr(v, 'shape') and len(getattr(v, 'shape', [])) > 0:
                                    bs = int(v.shape[0])
                                    break
                                elif isinstance(v, (list, tuple)) and len(v) > 0:
                                    bs = len(v)
                                    break
                        elif isinstance(b, dict):
                            for k, v in b.items():
                                if hasattr(v, 'shape') and len(getattr(v, 'shape', [])) > 0:
                                    bs = int(v.shape[0])
                                    break
                                elif isinstance(v, (list, tuple)) and len(v) > 0:
                                    bs = len(v)
                                    break
                    except Exception:
                        bs = 0
                    print(f"- Batch {i+1}: {dt:.3f}s (size={bs})")
                except StopIteration:
                    print(f"- Batch {i+1}: no more data")
                    break
                except Exception as e:
                    print(f"- Batch {i+1}: error {e}")
                    break
            
            try:
                batch = next(iter(train_loader))
                # RecBole возвращает Interaction объект, а не обычный dict
                if hasattr(batch, 'interaction'):
                    keys = list(batch.interaction.keys())
                    print(f"- Поля в батче: {keys}")
                    for key in keys:
                        value = batch.interaction[key]
                        if hasattr(value, 'shape'):
                            print(f"  {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"  {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"- Неожиданный тип батча: {type(batch)}")
            except Exception as e:
                print(f"- ⚠️ Ошибка при получении батча: {e}")

            # Измерим скорость итерирования по train_loader (ограничим до N батчей)
            def measure_loader_speed(loader, max_batches: int = 200):
                start = time.time()
                total_rows = 0
                num_batches = 0
                for i, b in enumerate(loader):
                    # Определяем размер батча по первому тензорному полю
                    bs = 0
                    try:
                        # RecBole возвращает Interaction объект
                        if hasattr(b, 'interaction'):
                            keys = list(b.interaction.keys())
                            # Попробуем сначала основные поля (user_id, item_id)
                            priority_keys = ['user_id', 'item_id', 'target', 'date']
                            check_keys = [k for k in priority_keys if k in keys] + [k for k in keys if k not in priority_keys]
                            
                            for k in check_keys:
                                v = b.interaction[k]
                                # Для torch.Tensor
                                if hasattr(v, 'shape') and len(getattr(v, 'shape', [])) > 0:
                                    bs = int(v.shape[0])
                                    break
                                # Для списков/кортежей
                                elif isinstance(v, (list, tuple)) and len(v) > 0:
                                    bs = len(v)
                                    break
                                # Для других объектов с __len__
                                elif hasattr(v, '__len__') and not isinstance(v, str):
                                    try:
                                        bs = len(v)
                                        if bs > 0:
                                            break
                                    except:
                                        pass
                        else:
                            # Fallback для обычных dict
                            keys = list(b.keys())
                            for k in keys:
                                v = b[k]
                                if hasattr(v, 'shape') and len(getattr(v, 'shape', [])) > 0:
                                    bs = int(v.shape[0])
                                    break
                    except Exception:
                        bs = 0
                    total_rows += bs
                    num_batches += 1
                    if max_batches is not None and (i + 1) >= max_batches:
                        break
                elapsed = time.time() - start
                ips = (total_rows / elapsed) if elapsed > 0 else 0.0
                bps = (num_batches / elapsed) if elapsed > 0 else 0.0
                return {
                    "batches": num_batches,
                    "rows": total_rows,
                    "seconds": elapsed,
                    "rows_per_sec": ips,
                    "batches_per_sec": bps,
                }

            stats = measure_loader_speed(train_loader, max_batches=200)
            print("\n🚀 Скорость итерации train_loader (первые 200 батчей):")
            print(f"- Batches: {stats['batches']}")
            print(f"- Rows: {stats['rows']}")
            print(f"- Time: {stats['seconds']:.3f}s")
            print(f"- Rows/sec: {stats['rows_per_sec']:.1f}")
            print(f"- Batches/sec: {stats['batches_per_sec']:.1f}")

        # Аналогично замерим valid_loader, если он есть
        valid_loader = valid_data.get("valid_loader")
        if valid_loader is not None:
            stats_v = measure_loader_speed(valid_loader, max_batches=200)
            print("\n🚀 Скорость итерации valid_loader (первые 200 батчей):")
            print(f"- Batches: {stats_v['batches']}")
            print(f"- Rows: {stats_v['rows']}")
            print(f"- Time: {stats_v['seconds']:.3f}s")
            print(f"- Rows/sec: {stats_v['rows_per_sec']:.1f}")
            print(f"- Batches/sec: {stats_v['batches_per_sec']:.1f}")
                
    except Exception as e:
        print(f"❌ Ошибка при выполнении prepare_dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

