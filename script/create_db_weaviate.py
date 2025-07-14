"""
Create a vector database using Weaviate
"""
import weaviate
import weaviate.classes as wvc

def main():

    # Initialize a local Weaviate instance
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
    )

    # Check whether metadata collection exists, if not, create schema for metadata
    if not client.collections.exists("Metadata"):
        metadata = client.collections.create(
            name="Metadata",
            description="Metadata of scientific papers",
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
            # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            properties=[
                wvc.Property(name="doi",
                             data_type=wvc.DataType.TEXT,
                             skip_vectorization=True,
                             index_filterable=True,
                             index_searchable=True),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="abstract",
                             data_type=wvc.DataType.TEXT)
            ]
        )
    else:
        print("Metadata collection exists.")
        metadata = client.collections.get("Metadata")
        pass

    if not client.collections.exists("FullText"):
        # Create schema for full text
        fulltext = client.collections.create(
            name="FullText",
            description="Full text of scientific papers",
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
            inverted_index_config=wvc.Configure.inverted_index(index_null_state=True),
            # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            properties=[
                wvc.Property(name="header_1", data_type=wvc.DataType.TEXT, index_searchable=True, description="level 1 "
                                                                                                              "header"),
                wvc.Property(name="header_2", data_type=wvc.DataType.TEXT, index_searchable=True, description="level 2 "
                                                                                                              "header"),
                wvc.Property(name="header_3", data_type=wvc.DataType.TEXT, index_searchable=True, description="level 3 "
                                                                                                              "header"),
                wvc.Property(name="header_4", data_type=wvc.DataType.TEXT, index_searchable=True, description="level 4 "
                                                                                                              "header"),
                wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                wvc.ReferenceProperty(name="hasMetadata", target_collection="Metadata")
            ]
        )
    else:
        print("FullText collection exists.")
        pass

    metadata.config.add_property(
        wvc.ReferenceProperty(
            name="hasFullText",
            target_collection="FullText"
        )
    )

    if not client.collections.exists("Table"):
        # Create schema for table
        table = client.collections.create(
            name="Table",
            description="Table of scientific papers",
            vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
            inverted_index_config=wvc.Configure.inverted_index(index_null_state=True),
            # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            properties=[
                wvc.Property(name="label", data_type=wvc.DataType.TEXT),
                wvc.Property(name="caption", data_type=wvc.DataType.TEXT),
                wvc.Property(name="tbody", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.ReferenceProperty(name="hasMetadata", target_collection="Metadata")
            ]
        )
    else:
        print("Table collection exists.")
        pass
    metadata = client.collections.get("Metadata")
    metadata.config.add_property(
        wvc.ReferenceProperty(
            name="hasTable",
            target_collection="Table"
        )
    )

    print("Schema created successfully!")


if __name__ == "__main__":
    main()