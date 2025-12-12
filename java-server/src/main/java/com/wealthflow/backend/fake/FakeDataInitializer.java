package com.wealthflow.backend.fake;

import com.wealthflow.backend.repository.AssetRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class FakeDataInitializer implements CommandLineRunner {

    private final AssetRepository assetRepository;

    public FakeDataInitializer(AssetRepository assetRepository) {
        this.assetRepository = assetRepository;
    }

    @Override
    public void run(String... args) {

        if (assetRepository.count() == 0) {
            assetRepository.saveAll(FakeAssetData.ASSETS);
        }
    }
}
