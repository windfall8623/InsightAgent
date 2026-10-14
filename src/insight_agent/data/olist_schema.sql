-- Kaggle Olist Brazilian E-Commerce Schema
-- Real-world 9-table relational database with indices and foreign constraints.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS olist_customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix TEXT NOT NULL,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS olist_sellers (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix TEXT NOT NULL,
    seller_city TEXT NOT NULL,
    seller_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS olist_products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_weight_g REAL,
    product_length_cm REAL,
    product_height_cm REAL,
    product_width_cm REAL
);

CREATE TABLE IF NOT EXISTS product_category_name_translation (
    product_category_name TEXT PRIMARY KEY,
    product_category_name_english TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS olist_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT NOT NULL,
    order_purchase_timestamp TEXT NOT NULL,
    order_approved_at TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES olist_customers(customer_id)
);

CREATE TABLE IF NOT EXISTS olist_order_items (
    order_id TEXT NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    shipping_limit_date TEXT NOT NULL,
    price REAL NOT NULL,
    freight_value REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id),
    FOREIGN KEY (product_id) REFERENCES olist_products(product_id),
    FOREIGN KEY (seller_id) REFERENCES olist_sellers(seller_id)
);

CREATE TABLE IF NOT EXISTS olist_order_payments (
    order_id TEXT NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_type TEXT NOT NULL,
    payment_installments INTEGER NOT NULL,
    payment_value REAL NOT NULL,
    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
);

CREATE TABLE IF NOT EXISTS olist_order_reviews (
    review_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    review_score INTEGER NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TEXT NOT NULL,
    review_answer_timestamp TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES olist_orders(order_id)
);

CREATE TABLE IF NOT EXISTS olist_geolocation (
    geolocation_zip_code_prefix TEXT NOT NULL,
    geolocation_lat REAL NOT NULL,
    geolocation_lng REAL NOT NULL,
    geolocation_city TEXT NOT NULL,
    geolocation_state TEXT NOT NULL
);

-- Indices for realistic production querying
CREATE INDEX IF NOT EXISTS idx_orders_customer ON olist_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_purchase_date ON olist_orders(order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status ON olist_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_items_product ON olist_order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_items_seller ON olist_order_items(seller_id);
CREATE INDEX IF NOT EXISTS idx_reviews_order ON olist_order_reviews(order_id);
CREATE INDEX IF NOT EXISTS idx_reviews_score ON olist_order_reviews(review_score);
CREATE INDEX IF NOT EXISTS idx_customers_state_city ON olist_customers(customer_state, customer_city);
CREATE INDEX IF NOT EXISTS idx_products_category ON olist_products(product_category_name);
