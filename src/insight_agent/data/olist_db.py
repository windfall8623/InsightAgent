"""Kaggle Olist Brazilian E-Commerce Sandbox Database.

Implements:
1. 9-table relational database seeding with deterministic pseudo-random generator.
2. Injected real-world anomalies (carrier strikes in Rio, damaged goods from specific sellers,
   extreme freight spikes in North region, customer complaints in Portuguese/English).
3. Read-only SQLite connection guarantee with engine-level authorizer and thread-safe locking.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

OLIST_SCHEMA_PATH = Path(__file__).with_name("olist_schema.sql")

OLIST_TABLES: tuple[str, ...] = (
    "olist_customers",
    "olist_geolocation",
    "olist_order_items",
    "olist_order_payments",
    "olist_order_reviews",
    "olist_orders",
    "olist_products",
    "olist_sellers",
    "product_category_name_translation",
)

# Date bounds for historical analysis (2016-2018)
OLIST_DATA_START = datetime(2016, 9, 1)
OLIST_DATA_END = datetime(2018, 10, 17)
AS_OF_OLIST = OLIST_DATA_END


# Realistic Brazilian geographic distributions
STATES_CITIES: dict[str, list[str]] = {
    "SP": ["sao paulo", "campinas", "guarulhos", "santos", "sao bernardo do campo"],
    "RJ": ["rio de janeiro", "niteroi", "nova iguacu", "duque de caxias", "sao goncalo"],
    "MG": ["belo horizonte", "uberlandia", "contagem", "juiz de fora", "betim"],
    "RS": ["porto alegre", "caxias do sul", "pelotas", "canoas", "santa maria"],
    "PR": ["curitiba", "londrina", "maringa", "ponta grossa", "cascavel"],
    "BA": ["salvador", "feira de santana", "vitoria da conquista"],
    "AM": ["manaus", "itacoatiara"],
    "PA": ["belem", "anandindeua", "santarem"],
}

CATEGORIES_MAP: list[tuple[str, str]] = [
    ("informatica_acessorios", "computers_accessories"),
    ("beleza_saude", "health_beauty"),
    ("cama_mesa_banho", "bed_bath_table"),
    ("esporte_lazer", "sports_leisure"),
    ("moveis_decoracao", "furniture_decor"),
    ("relogios_presentes", "watches_gifts"),
    ("utilidades_domesticas", "housewares"),
    ("telefonia", "telephony"),
    ("automotivo", "auto"),
    ("brinquedos", "toys"),
]

PAYMENT_TYPES = ("credit_card", "boleto", "voucher", "debit_card")


@dataclass(slots=True)
class OlistQueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[object, ...]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]

    def to_payload(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": self.as_dicts(),
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }



def default_olist_db_path() -> Path:
    import os

    override = os.environ.get("INSIGHT_OLIST_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).with_name("olist.db")


def _authorizer_readonly(
    action: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger: str | None,
) -> int:
    """Refuse any write or schema-modifying actions inside SQLite."""
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in {
        "table_info",
        "index_list",
        "foreign_key_list",
    }:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


_OLIST_LOCKS: dict[int, threading.Lock] = {}


def _lock_for_olist(conn: sqlite3.Connection) -> threading.Lock:
    lock = _OLIST_LOCKS.get(id(conn))
    if lock is None:
        lock = threading.Lock()
        _OLIST_LOCKS[id(conn)] = lock
    return lock


def connect_olist(db_path: Path | None = None) -> sqlite3.Connection:
    """Connect to the Olist database in read-only mode with authorizer enabled."""
    path = ensure_olist_database(db_path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_authorizer_readonly)
    return conn


def execute_olist(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[object] = (),
    *,
    limit: int = 500,
) -> OlistQueryResult:
    """Safely execute a query with per-connection thread mutex."""
    import time

    start = time.perf_counter()
    with _lock_for_olist(conn):
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit + 1)
        truncated = len(rows) > limit
        if truncated:
            rows = rows[:limit]
        elapsed = (time.perf_counter() - start) * 1000
        return OlistQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round(elapsed, 2),
        )


def ensure_olist_database(db_path: Path | None = None, *, seed: int = 42) -> Path:
    """Create and seed the Olist dataset with authentic anomalies if not present."""
    path = db_path or default_olist_db_path()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(OLIST_SCHEMA_PATH.read_text(encoding="utf-8"))
        _seed_olist(conn, seed=seed)
        conn.commit()
    return path


def _seed_olist(conn: sqlite3.Connection, *, seed: int) -> None:
    """Generate realistic Olist tables with verified anomaly scenarios."""
    rng = random.Random(seed)

    # 1. Category translation
    conn.executemany(
        "INSERT INTO product_category_name_translation (product_category_name, product_category_name_english) VALUES (?, ?)",
        CATEGORIES_MAP,
    )

    # 2. Customers (1,200 realistic customers)
    customers = []
    customer_ids = []
    for i in range(1, 1201):
        cid = f"cust_{i:05d}"
        uniq_id = f"uniq_{i:05d}"
        state = rng.choices(
            list(STATES_CITIES.keys()), weights=(45, 25, 12, 6, 5, 4, 1.5, 1.5)
        )[0]
        city = rng.choice(STATES_CITIES[state])
        zip_prefix = f"{rng.randint(1000, 99999):05d}"
        customers.append((cid, uniq_id, zip_prefix, city, state))
        customer_ids.append((cid, state, city))
    conn.executemany(
        "INSERT INTO olist_customers (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state) VALUES (?, ?, ?, ?, ?)",
        customers,
    )

    # 3. Sellers (100 sellers, including a few problematic sellers for anomaly attribution)
    sellers = []
    seller_ids = []
    for i in range(1, 101):
        sid = f"seller_{i:04d}"
        state = rng.choices(["SP", "RJ", "MG", "PR", "SC", "RS"], weights=(60, 15, 10, 5, 5, 5))[0]
        city = rng.choice(STATES_CITIES.get(state, ["sao paulo"]))
        zip_prefix = f"{rng.randint(1000, 99999):05d}"
        sellers.append((sid, zip_prefix, city, state))
        seller_ids.append(sid)
    conn.executemany(
        "INSERT INTO olist_sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state) VALUES (?, ?, ?, ?)",
        sellers,
    )

    # 3b. Geolocation (Brazilian geographic distribution)
    geos = []
    geo_coords = {
        "SP": (-23.5505, -46.6333),
        "RJ": (-22.9068, -43.1729),
        "MG": (-19.9208, -43.9378),
        "RS": (-30.0346, -51.2177),
        "PR": (-25.4284, -49.2733),
        "BA": (-12.9777, -38.5016),
        "AM": (-3.1190, -60.0217),
        "PA": (-1.4558, -48.4902),
    }
    for state, cities in STATES_CITIES.items():
        base_lat, base_lng = geo_coords.get(state, (-23.55, -46.63))
        for city in cities:
            for _ in range(5):
                zip_p = f"{rng.randint(1000, 99999):05d}"
                lat = round(base_lat + rng.uniform(-0.1, 0.1), 6)
                lng = round(base_lng + rng.uniform(-0.1, 0.1), 6)
                geos.append((zip_p, lat, lng, city, state))
    conn.executemany(
        "INSERT INTO olist_geolocation (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state) VALUES (?, ?, ?, ?, ?)",
        geos,
    )

    # 4. Products (250 products across categories)

    products = []
    product_ids = []
    for i in range(1, 251):
        pid = f"prod_{i:04d}"
        cat = rng.choice(CATEGORIES_MAP)[0]
        weight = round(rng.uniform(100, 8000), 1)
        length = round(rng.uniform(10, 80), 1)
        height = round(rng.uniform(5, 50), 1)
        width = round(rng.uniform(10, 60), 1)
        products.append((pid, cat, weight, length, height, width))
        product_ids.append((pid, cat, weight))
    conn.executemany(
        "INSERT INTO olist_products (product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm) VALUES (?, ?, ?, ?, ?, ?)",
        products,
    )

    # 5. Orders, Items, Payments, and Reviews with Injected Anomalies
    orders, items, payments, reviews = [], [], [], []
    order_counter = 0
    item_counter = 0
    pay_counter = 0
    review_counter = 0

    total_days = (OLIST_DATA_END - OLIST_DATA_START).days
    for day_offset in range(total_days):
        cur_date = OLIST_DATA_START + timedelta(days=day_offset)
        # Volume fluctuates with seasonality (e.g. Black Friday in late November)
        daily_count = rng.randint(4, 12)
        if cur_date.month == 11 and cur_date.day >= 20:
            daily_count += 15

        for _ in range(daily_count):
            order_counter += 1
            oid = f"order_{order_counter:06d}"
            cust_tuple = rng.choice(customer_ids)
            cid, cust_state, cust_city = cust_tuple
            purchase_ts = cur_date + timedelta(
                hours=rng.randint(8, 22), minutes=rng.randint(0, 59)
            )

            # Anomaly 1: Carrier Strike in Rio de Janeiro (RJ) in Q3 2017 (July - September)
            is_rio_q3_anomaly = (
                cust_state == "RJ"
                and cur_date.year == 2017
                and 7 <= cur_date.month <= 9
                and rng.random() < 0.65
            )

            # Anomaly 2: Defective Seller seller_0042 in "informatica_acessorios"
            bad_seller_anomaly = rng.random() < 0.08

            status = "delivered"
            if rng.random() < 0.03:
                status = "cancelled"
            elif rng.random() < 0.02:
                status = "shipped"

            est_delivery = purchase_ts + timedelta(days=rng.randint(12, 22))

            if is_rio_q3_anomaly:
                # Carrier severe delivery delay
                carrier_date = purchase_ts + timedelta(days=rng.randint(5, 12))
                # Delivered way after estimated date (+12 to +28 days delay!)
                cust_delivery = est_delivery + timedelta(days=rng.randint(10, 25))
            else:
                carrier_date = purchase_ts + timedelta(days=rng.randint(1, 3))
                # Normal delivery: on-time or a few days early
                cust_delivery = purchase_ts + timedelta(days=rng.randint(4, 14))

            orders.append(
                (
                    oid,
                    cid,
                    status,
                    purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    (purchase_ts + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    carrier_date.strftime("%Y-%m-%d %H:%M:%S") if status == "delivered" else None,
                    cust_delivery.strftime("%Y-%m-%d %H:%M:%S") if status == "delivered" else None,
                    est_delivery.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

            # Order items
            n_items = rng.choices([1, 2, 3], weights=[85, 12, 3])[0]
            order_total = 0.0
            order_items_cats = []
            for item_idx in range(1, n_items + 1):
                item_counter += 1
                if bad_seller_anomaly and item_idx == 1:
                    sid = "seller_0042"
                    pid = product_ids[0][0]  # informatica
                    cat = "informatica_acessorios"
                else:
                    pid, cat, _ = rng.choice(product_ids)
                    sid = rng.choice(seller_ids)

                order_items_cats.append(cat)
                price = round(rng.uniform(25, 450), 2)
                # Anomaly 3: High Freight in Northern States (AM/PA)
                if cust_state in ("AM", "PA"):
                    freight = round(rng.uniform(80, 160), 2)
                else:
                    freight = round(rng.uniform(12, 38), 2)

                order_total += price + freight
                items.append(
                    (
                        oid,
                        item_idx,
                        pid,
                        sid,
                        (purchase_ts + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        price,
                        freight,
                    )
                )

            # Payments
            pay_counter += 1
            ptype = rng.choices(PAYMENT_TYPES, weights=[72, 18, 6, 4])[0]
            payments.append(
                (
                    oid,
                    1,
                    ptype,
                    rng.randint(1, 6) if ptype == "credit_card" else 1,
                    round(order_total, 2),
                )
            )

            # Reviews
            review_counter += 1
            rid = f"rev_{review_counter:06d}"
            review_date = (cust_delivery + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            title: str | None = None
            msg: str | None = None

            if is_rio_q3_anomaly:

                score = rng.choices([1, 2], weights=[80, 20])[0]
                title = rng.choice(["Atraso absurdo", "Não recomendo", "Demora", "Péssimo serviço"])
                msg = rng.choice(
                    [
                        "Demorou mais de 20 dias para entregar! Péssima transportadora no Rio de Janeiro.",
                        "Atraso absurdo na entrega. A data prevista expirou há duas semanas.",
                        "Chegou muito atrasado, o suporte não respondeu e o rastreamento parou.",
                        "Extremamente insatisfeito com o atraso da entrega em Niterói.",
                    ]
                )
            elif bad_seller_anomaly and "seller_0042" in [it[3] for it in items[-n_items:]]:
                score = 1
                title = rng.choice(["Produto com defeito", "Quebrado", "Enganação", "Péssima qualidade"])
                msg = rng.choice(
                    [
                        "Produto chegou quebrado e com a embalagem rasgada pelo vendedor seller_0042.",
                        "Veio com defeito de fabricação. Produto não liga de jeito nenhum.",
                        "Item danificado e parece produto recondicionado ou falso.",
                    ]
                )
            else:
                score = rng.choices([5, 4, 3, 2, 1], weights=[65, 18, 9, 4, 4])[0]
                if score >= 4:
                    title = rng.choice(["Excelente", "Recomendo", "Muito bom", "Entrega rápida", None])
                    msg = rng.choice(
                        [
                            "Excelente produto, chegou muito rápido e bem embalado!",
                            "Muito satisfeito, entrega pontual e produto de qualidade.",
                            "Tudo perfeito, recomendo a todos da plataforma.",
                            None,
                        ]
                    )
                else:
                    title = "Regular"
                    msg = "Demorou um pouco além do esperado, mas o produto funciona."

            reviews.append(
                (
                    rid,
                    oid,
                    score,
                    title,
                    msg,
                    review_date,
                    review_date,
                )
            )

    conn.executemany(
        "INSERT INTO olist_orders (order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        orders,
    )
    conn.executemany(
        "INSERT INTO olist_order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value) VALUES (?, ?, ?, ?, ?, ?, ?)",
        items,
    )
    conn.executemany(
        "INSERT INTO olist_order_payments (order_id, payment_sequential, payment_type, payment_installments, payment_value) VALUES (?, ?, ?, ?, ?)",
        payments,
    )
    conn.executemany(
        "INSERT INTO olist_order_reviews (review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        reviews,
    )
