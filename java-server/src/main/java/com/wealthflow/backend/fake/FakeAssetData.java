package com.wealthflow.backend.fake;

import com.wealthflow.backend.model.Asset;
import com.wealthflow.backend.model.enums.AssetType;
import com.wealthflow.backend.model.enums.RiskLevel;

import java.util.List;

public final class FakeAssetData {

    private FakeAssetData() {}

    public static final List<Asset> ASSETS = List.of(

            // ===== STOCKS =====
            new Asset("AAPL", "Apple Inc.", AssetType.STOCK, RiskLevel.MEDIUM, 8.7),
            new Asset("MSFT", "Microsoft Corp.", AssetType.STOCK, RiskLevel.MEDIUM, 8.6),
            new Asset("GOOGL", "Alphabet Inc.", AssetType.STOCK, RiskLevel.MEDIUM, 7.9),
            new Asset("AMZN", "Amazon.com Inc.", AssetType.STOCK, RiskLevel.MEDIUM, 6.8),
            new Asset("META", "Meta Platforms Inc.", AssetType.STOCK, RiskLevel.MEDIUM, 6.5),
            new Asset("NVDA", "NVIDIA Corp.", AssetType.STOCK, RiskLevel.HIGH, 7.4),
            new Asset("TSLA", "Tesla Inc.", AssetType.STOCK, RiskLevel.HIGH, 7.1),
            new Asset("JPM", "JPMorgan Chase", AssetType.STOCK, RiskLevel.LOW, 6.9),
            new Asset("V", "Visa Inc.", AssetType.STOCK, RiskLevel.LOW, 7.3),
            new Asset("KO", "Coca-Cola Co.", AssetType.STOCK, RiskLevel.LOW, 8.1),
            new Asset("PEP", "PepsiCo Inc.", AssetType.STOCK, RiskLevel.LOW, 8.0),
            new Asset("WMT", "Walmart Inc.", AssetType.STOCK, RiskLevel.LOW, 7.6),
            new Asset("NKE", "Nike Inc.", AssetType.STOCK, RiskLevel.MEDIUM, 7.2),
            new Asset("DIS", "Walt Disney Co.", AssetType.STOCK, RiskLevel.MEDIUM, 6.9),
            new Asset("INTC", "Intel Corp.", AssetType.STOCK, RiskLevel.MEDIUM, 6.4),

            // ===== ETFS =====
            new Asset("SPY", "S&P 500 ETF", AssetType.ETF, RiskLevel.MEDIUM, 7.6),
            new Asset("QQQ", "NASDAQ 100 ETF", AssetType.ETF, RiskLevel.MEDIUM, 7.2),
            new Asset("VTI", "Total Stock Market ETF", AssetType.ETF, RiskLevel.LOW, 7.9),
            new Asset("VOO", "Vanguard S&P 500 ETF", AssetType.ETF, RiskLevel.LOW, 7.8),
            new Asset("ARKK", "ARK Innovation ETF", AssetType.ETF, RiskLevel.HIGH, 6.1),
            new Asset("IWM", "Russell 2000 ETF", AssetType.ETF, RiskLevel.MEDIUM, 6.8),
            new Asset("EEM", "Emerging Markets ETF", AssetType.ETF, RiskLevel.HIGH, 6.3),

            // ===== BONDS =====
            new Asset("TLT", "20+ Year Treasury Bond ETF", AssetType.BOND, RiskLevel.LOW, 8.2),
            new Asset("IEF", "7–10 Year Treasury Bond ETF", AssetType.BOND, RiskLevel.LOW, 8.4),
            new Asset("LQD", "Investment Grade Corporate Bonds", AssetType.BOND, RiskLevel.LOW, 7.8),
            new Asset("BND", "Total Bond Market ETF", AssetType.BOND, RiskLevel.LOW, 8.1),
            new Asset("AGG", "Core US Aggregate Bond ETF", AssetType.BOND, RiskLevel.LOW, 8.0),

            // ===== COMMODITIES =====
            new Asset("GLD", "Gold Trust ETF", AssetType.COMMODITY, RiskLevel.LOW, 7.5),
            new Asset("SLV", "Silver Trust ETF", AssetType.COMMODITY, RiskLevel.MEDIUM, 6.9),
            new Asset("USO", "US Oil Fund", AssetType.COMMODITY, RiskLevel.HIGH, 5.4),
            new Asset("DBA", "Agriculture Commodities ETF", AssetType.COMMODITY, RiskLevel.MEDIUM, 6.7),

            // ===== CRYPTO =====
            new Asset("BTC", "Bitcoin", AssetType.CRYPTO, RiskLevel.HIGH, 4.3),
            new Asset("ETH", "Ethereum", AssetType.CRYPTO, RiskLevel.HIGH, 5.1),
            new Asset("SOL", "Solana", AssetType.CRYPTO, RiskLevel.HIGH, 4.8),
            new Asset("ADA", "Cardano", AssetType.CRYPTO, RiskLevel.HIGH, 4.9),
            new Asset("AVAX", "Avalanche", AssetType.CRYPTO, RiskLevel.HIGH, 4.6),
            new Asset("DOT", "Polkadot", AssetType.CRYPTO, RiskLevel.HIGH, 4.7),

            // ===== GLOBAL / EXTRA =====
            new Asset("SAP", "SAP SE", AssetType.STOCK, RiskLevel.LOW, 7.4),
            new Asset("NESN", "Nestlé SA", AssetType.STOCK, RiskLevel.LOW, 8.3),
            new Asset("TM", "Toyota Motor Corp.", AssetType.STOCK, RiskLevel.LOW, 7.8),
            new Asset("BABA", "Alibaba Group", AssetType.STOCK, RiskLevel.HIGH, 5.9),
            new Asset("SONY", "Sony Group Corp.", AssetType.STOCK, RiskLevel.MEDIUM, 7.1),

            new Asset("XLF", "Financial Sector ETF", AssetType.ETF, RiskLevel.MEDIUM, 6.9),
            new Asset("XLK", "Technology Sector ETF", AssetType.ETF, RiskLevel.MEDIUM, 7.0),
            new Asset("XLV", "Healthcare Sector ETF", AssetType.ETF, RiskLevel.LOW, 7.7),

            new Asset("REIT", "US Real Estate ETF", AssetType.ETF, RiskLevel.MEDIUM, 6.8),
            new Asset("VNQ", "Vanguard Real Estate ETF", AssetType.ETF, RiskLevel.MEDIUM, 7.1),

            new Asset("COP", "ConocoPhillips", AssetType.STOCK, RiskLevel.HIGH, 5.8),
            new Asset("XOM", "Exxon Mobil Corp.", AssetType.STOCK, RiskLevel.MEDIUM, 5.6),
            new Asset("BP", "BP plc", AssetType.STOCK, RiskLevel.MEDIUM, 6.0)
    );
}
