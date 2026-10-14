-- E-commerce sandbox schema.
-- Mirrors a conventional OLTP commerce model so the agent's queries look like
-- production queries, not toy tables.

DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    city          TEXT    NOT NULL,
    register_date TEXT    NOT NULL,
    member_level  TEXT    NOT NULL CHECK (member_level IN ('bronze', 'silver', 'gold'))
);

CREATE TABLE categories (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,
    parent_id INTEGER REFERENCES categories(id)
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    price       REAL    NOT NULL CHECK (price >= 0),
    stock       INTEGER NOT NULL CHECK (stock >= 0)
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    order_date   TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('created', 'paid', 'shipped', 'completed', 'cancelled')),
    total_amount REAL    NOT NULL,
    channel      TEXT    NOT NULL CHECK (channel IN ('app', 'web', 'mini_program'))
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL    NOT NULL
);

CREATE TABLE payments (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    pay_method TEXT    NOT NULL CHECK (pay_method IN ('alipay', 'wechat', 'card')),
    pay_amount REAL    NOT NULL,
    pay_status TEXT    NOT NULL CHECK (pay_status IN ('success', 'failed', 'pending'))
);

CREATE TABLE refunds (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    amount      REAL    NOT NULL,
    reason      TEXT    NOT NULL,
    refund_date TEXT    NOT NULL
);

-- Column and table access patterns the agent's queries hit most.
CREATE INDEX idx_orders_date     ON orders(order_date);
CREATE INDEX idx_orders_user     ON orders(user_id);
CREATE INDEX idx_orders_status   ON orders(status);
CREATE INDEX idx_users_city      ON users(city);
CREATE INDEX idx_products_cat    ON products(category_id);
CREATE INDEX idx_items_order     ON order_items(order_id);
CREATE INDEX idx_payments_order  ON payments(order_id);
CREATE INDEX idx_refunds_order   ON refunds(order_id);