var PriceItPreprocessor = function() {};

PriceItPreprocessor.prototype = {
  run: function(arguments) {
    const payload = this.extractProductData();
    arguments.completionFunction(payload);
  },

  finalize: function(arguments) {
    return;
  },

  extractProductData: function() {
    const data = {
      product_name: null,
      current_price: null,
      currency: null,
      brand: null,
      gtin13: null,
      gtin8: null,
      sku: null,
      source_url: document.location.href
    };

    const ldData = this.extractFromJsonLd();
    if (ldData) {
      Object.assign(data, ldData);
    }

    if (!data.product_name) {
      data.product_name = this.getMetaContent('property', 'og:title') || this.getMetaContent('name', 'title');
    }

    if (!data.current_price) {
      data.current_price = this.getMetaContent('property', 'product:price:amount') ||
                           this.getMetaContent('property', 'og:price:amount') ||
                           this.getMetaContent('name', 'price');
    }

    if (!data.currency) {
      data.currency = this.getMetaContent('property', 'product:price:currency') ||
                      this.getMetaContent('property', 'og:price:currency') ||
                      this.getMetaContent('name', 'currency');
    }

    if (!data.brand) {
      data.brand = this.getMetaContent('name', 'brand') || this.getMetaContent('property', 'product:brand');
    }

    return data;
  },

  extractFromJsonLd: function() {
    const scripts = document.querySelectorAll('script[type="application/ld+json"]');

    for (let i = 0; i < scripts.length; i++) {
      const text = scripts[i].textContent;
      if (!text) continue;

      try {
        const parsed = JSON.parse(text);
        const product = this.findProductNode(parsed);
        if (!product) continue;

        const offers = Array.isArray(product.offers) ? product.offers[0] : product.offers || {};
        const brand = typeof product.brand === 'string'
          ? product.brand
          : (product.brand && product.brand.name) ? product.brand.name : null;

        return {
          product_name: product.name || null,
          current_price: offers.price || product.price || null,
          currency: offers.priceCurrency || product.priceCurrency || null,
          brand: brand,
          gtin13: product.gtin13 || null,
          gtin8: product.gtin8 || null,
          sku: product.sku || null
        };
      } catch (e) {
        // Ignore malformed JSON-LD and continue.
      }
    }

    return null;
  },

  findProductNode: function(node) {
    if (!node) return null;

    if (Array.isArray(node)) {
      for (let i = 0; i < node.length; i++) {
        const found = this.findProductNode(node[i]);
        if (found) return found;
      }
      return null;
    }

    if (typeof node === 'object') {
      const type = node['@type'];
      if ((typeof type === 'string' && type.toLowerCase() === 'product') ||
          (Array.isArray(type) && type.some(t => String(t).toLowerCase() === 'product'))) {
        return node;
      }

      if (node['@graph']) {
        const foundGraph = this.findProductNode(node['@graph']);
        if (foundGraph) return foundGraph;
      }
    }

    return null;
  },

  getMetaContent: function(attrName, attrValue) {
    const selector = `meta[${attrName}="${attrValue}"]`;
    const el = document.querySelector(selector);
    if (!el) return null;
    return el.getAttribute('content');
  }
};

var ExtensionPreprocessingJS = new PriceItPreprocessor();
