-- CN-Retail: Domestic E-Commerce Retail Data Warehouse Schema
-- 7 core high-cohesion tables modeling Chinese e-commerce trading, fulfillment, reviews, and aftersales.

PRAGMA foreign_keys = ON;

-- 1. 买家用户表
CREATE TABLE IF NOT EXISTS cn_users (
    user_id TEXT PRIMARY KEY,
    province TEXT NOT NULL,
    city TEXT NOT NULL,
    member_level TEXT NOT NULL -- 普客 / 银卡会员 / 金卡会员 / 黑钻VIP
);

-- 2. 商家店铺表
CREATE TABLE IF NOT EXISTS cn_sellers (
    seller_id TEXT PRIMARY KEY,
    seller_name TEXT NOT NULL,
    province TEXT NOT NULL,
    city TEXT NOT NULL
);

-- 3. 商品库 (包含一级类目与二级类目)
CREATE TABLE IF NOT EXISTS cn_products (
    product_id TEXT PRIMARY KEY,
    category_l1 TEXT NOT NULL, -- 生鲜食品 / 美妆个护 / 数码3C / 服饰鞋包 / 母婴用品 / 家家居装
    category_l2 TEXT NOT NULL, -- 冷冻海鲜 / 新鲜水果 / 面部精华 / 手机配件 / 智能数码等
    product_name TEXT NOT NULL,
    base_price REAL NOT NULL
);

-- 4. 订单事实表
CREATE TABLE IF NOT EXISTS cn_orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    order_status TEXT NOT NULL, -- 已妥投 / 运输中 / 已取消 / 售后中
    order_time TEXT NOT NULL,
    total_amount REAL NOT NULL, -- 实付总金额 (元)
    channel TEXT NOT NULL,      -- 天猫App / 微信小程序 / 直播间 / 淘宝网页版
    FOREIGN KEY (user_id) REFERENCES cn_users(user_id)
);

-- 5. 订单明细表 (拆单与商品关联)
CREATE TABLE IF NOT EXISTS cn_order_items (
    item_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES cn_orders(order_id),
    FOREIGN KEY (product_id) REFERENCES cn_products(product_id),
    FOREIGN KEY (seller_id) REFERENCES cn_sellers(seller_id)
);

-- 6. 物流履约表 (顺丰/中通/京东/圆通等时效跟踪)
CREATE TABLE IF NOT EXISTS cn_logistics (
    logistics_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    carrier TEXT NOT NULL,                -- 顺丰速运 / 中通快递 / 京东物流 / 圆通速递 / 极兔速递
    shipped_time TEXT,
    promise_delivery_time TEXT NOT NULL, -- 承诺最迟送达时间
    actual_delivery_time TEXT,           -- 实际送达时间
    route_status TEXT NOT NULL,          -- 正常在途 / 网点积压 / 天气受阻 / 已签收
    FOREIGN KEY (order_id) REFERENCES cn_orders(order_id)
);

-- 7. 买家评价表 (结构化打分与纯正中文追评)
CREATE TABLE IF NOT EXISTS cn_reviews (
    review_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    rating INTEGER NOT NULL,             -- 1 ~ 5 星
    tags TEXT,                           -- 差评标签 (时效迟缓 / 外包装挤压破损 / 货不对板 / 优质推荐)
    comment_text TEXT NOT NULL,          -- 中文买家评价全文
    FOREIGN KEY (order_id) REFERENCES cn_orders(order_id)
);

-- 8. 售后退款表
CREATE TABLE IF NOT EXISTS cn_aftersales (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    refund_type TEXT NOT NULL,           -- 仅退款 / 退货退款
    refund_reason TEXT NOT NULL,         -- 未按约定时间发货 / 商品质量问题 / 外包装破损 / 不喜欢/效果差
    refund_amount REAL NOT NULL,         -- 退款金额 (元)
    status TEXT NOT NULL,                -- 已退款 / 审核中 / 已驳回
    FOREIGN KEY (order_id) REFERENCES cn_orders(order_id)
);

-- 生产级高性能索引
CREATE INDEX IF NOT EXISTS idx_cn_orders_user ON cn_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_cn_orders_time ON cn_orders(order_time);
CREATE INDEX IF NOT EXISTS idx_cn_orders_status ON cn_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_cn_items_order ON cn_order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_cn_items_product ON cn_order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_cn_items_seller ON cn_order_items(seller_id);
CREATE INDEX IF NOT EXISTS idx_cn_logistics_order ON cn_logistics(order_id);
CREATE INDEX IF NOT EXISTS idx_cn_logistics_carrier ON cn_logistics(carrier);
CREATE INDEX IF NOT EXISTS idx_cn_logistics_status ON cn_logistics(route_status);
CREATE INDEX IF NOT EXISTS idx_cn_reviews_order ON cn_reviews(order_id);
CREATE INDEX IF NOT EXISTS idx_cn_reviews_rating ON cn_reviews(rating);
CREATE INDEX IF NOT EXISTS idx_cn_aftersales_order ON cn_aftersales(order_id);
CREATE INDEX IF NOT EXISTS idx_cn_users_city ON cn_users(city, province);
CREATE INDEX IF NOT EXISTS idx_cn_products_cat ON cn_products(category_l1, category_l2);
