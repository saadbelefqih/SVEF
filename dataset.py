class GroundTruthConstructor:
    """Construct curated ground truth schemas for datasets"""

    def __init__(self):
        self.schemas = {}

    def construct_ecommerce_ground_truth(self) -> Dict:
        """Curated e-commerce schema with known complexity patterns"""
        schema =  {
            'name': 'ECommerce',
            'complexity': 'high',
            'heterogeneity': 'medium',
            'evolution_patterns': ['property_addition', 'type_migration'],
            'entities': {
                'user': {
                    'properties': {
                        'user_id': {'type': 'string', 'required': True},
                        'name': {'type': 'string', 'required': True},
                        'email': {'type': 'string', 'required': True},
                        'age': {'type': 'integer', 'required': False},
                        'premium': {'type': 'boolean', 'required': False}
                    }
                },
                'order': {
                    'properties': {
                        'order_id': {'type': 'string', 'required': True},
                        'user_id': {'type': 'string', 'required': True, 'reference': 'user'},
                        'timestamp': {'type': 'string', 'required': True},
                        'total_amount': {'type': 'union', 'types': ['float', 'string'], 'required': True},
                        'status': {'type': 'string', 'required': True}
                    }
                }
            },
            'optional_properties': ['age', 'premium', 'loyalty_points'],
            'property_dependencies': {
                'premium': ['age']  # Premium membership requires age
            },
            'union_types': {
                'total_amount': {'float', 'string'},
                'metadata': {'object', 'string'}
            },
            'array_properties': {
                'order_items': {'dimensions': 1, 'element_type': 'object'},
                'tags': {'dimensions': 1, 'element_type': 'string'}
            },
            'relationships': {
                'user_orders': {
                    'type': 'reference',
                    'cardinality': '1:N',
                    'source': 'user',
                    'target': 'order'
                }
            }
        }
            # Add consistent evolution patterns structure
        schema['evolution_patterns'] = {
        'version_boundaries': ['2024-01-01', '2024-06-01'],
        'patterns': {
            'property_addition': {
                'properties_added': ['loyalty_points'],
                'timestamp': '2024-03-01'
            },
            'type_migration': {
                'property': 'total_amount',
                'from': 'string',
                'to': 'float',
                'timestamp': '2024-06-01'
            }
        },
        'version_sequence': ['v1', 'v2', 'v3'],
        'schema_drifts': ['loyalty_points_addition']
        }

        return schema

    def construct_healthcare_ground_truth(self) -> Dict:
        """Curated healthcare schema with high heterogeneity"""
        return {
            'name': 'Healthcare',
            'complexity': 'very_high',
            'heterogeneity': 'high',
            'evolution_patterns': ['property_removal', 'type_migration', 'schema_merge'],
            'entities': {
                'patient': {
                    'properties': {
                        'patient_id': {'type': 'string', 'required': True},
                        'name': {'type': 'string', 'required': True},
                        'birth_date': {'type': 'string', 'required': True},
                        'medical_history': {'type': 'array', 'required': False, 'element_type': 'object'},
                        'emergency_contact': {'type': 'union', 'types': ['object', 'string'], 'required': False}
                    }
                },
                'treatment': {
                    'properties': {
                        'treatment_id': {'type': 'string', 'required': True},
                        'patient_id': {'type': 'string', 'required': True, 'reference': 'patient'},
                        'medications': {'type': 'array', 'required': True, 'element_type': 'object'},
                        'dosage': {'type': 'union', 'types': ['float', 'string', 'object'], 'required': True}
                    }
                }
            },
            'optional_properties': ['medical_history', 'emergency_contact', 'allergies'],
            'union_types': {
                'dosage': {'float', 'string', 'object'},
                'emergency_contact': {'object', 'string'},
                'test_results': {'array', 'object', 'string'}
            },
            'complex_arrays': {
                'medical_history': {'dimensions': 1, 'heterogeneous_elements': True},
                'medications': {'dimensions': 1, 'nested_objects': True}
            },
            'temporal_evolution': {
                'versions': ['v1', 'v2', 'v3'],
                'changes': [
                    {'version': 'v2', 'change': 'property_addition', 'property': 'allergies'},
                    {'version': 'v3', 'change': 'type_migration', 'property': 'dosage', 'from': 'string', 'to': 'object'}
                ]
            }
        }

    def construct_iot_ground_truth(self) -> Dict:
        """Curated IoT schema with temporal evolution patterns"""
        return {
          'name': 'IoT',
          'complexity': 'medium',
          'heterogeneity': 'low',
          'entities': {
              'sensor': {
                  'properties': {
                      'sensor_id': {'type': 'string', 'required': True},
                      'location': {'type': 'object', 'required': True},
                      'readings': {'type': 'array', 'required': True, 'element_type': 'object'},
                      'timestamp': {'type': 'string', 'required': True}
                  }
              }
          },
          'evolution_patterns': {
              'version_boundaries': ['2024-01-01', '2024-06-01'],
              'patterns': {
                  'property_addition': True,
                  'schema_split': True
              },
              'version_sequence': ['v1', 'v2'],
              'schema_drifts': ['battery_addition']
          }
    }

class DatasetGenerator:
    """Generate reproducible synthetic datasets based on ground truth schemas"""

    def __init__(self, seed=42):
        self.random_state = np.random.RandomState(seed)

    def generate_ecommerce_data(self, num_documents=500) -> List[Dict]:
        """Generate e-commerce dataset with controlled heterogeneity"""
        documents = []

        for i in range(num_documents):
            doc_type = self.random_state.choice(['user', 'order'], p=[0.4, 0.6])

            if doc_type == 'user':
                doc = {
                    'user_id': f'user_{i:04d}',
                    'name': f'User {i}',
                    'email': f'user{i}@example.com'
                }

                # Controlled optionality
                if self.random_state.random() > 0.3:
                    doc['age'] = self.random_state.randint(18, 80)
                if self.random_state.random() > 0.7 and 'age' in doc:
                    doc['premium'] = self.random_state.choice([True, False])

            elif doc_type == 'order':
                doc = {
                    'order_id': f'order_{i:05d}',
                    'user_id': f'user_{self.random_state.randint(0, 200):04d}',
                    'timestamp': f'2024-01-{self.random_state.randint(1,28):02d}T10:00:00Z',
                    'status': self.random_state.choice(['pending', 'shipped', 'delivered'])
                }

                # Controlled type heterogeneity
                amount_type = self.random_state.choice(['float', 'string'], p=[0.7, 0.3])
                if amount_type == 'float':
                    doc['total_amount'] = round(self.random_state.uniform(10, 500), 2)
                else:
                    doc['total_amount'] = f"${round(self.random_state.uniform(10, 500), 2)}"

            documents.append(doc)

        return documents

    def generate_healthcare_data(self, num_documents=300) -> List[Dict]:
        """Generate healthcare dataset with high heterogeneity"""
        documents = []

        for i in range(num_documents):
            doc_type = self.random_state.choice(['patient', 'treatment'], p=[0.6, 0.4])

            if doc_type == 'patient':
                doc = {
                    'patient_id': f'pat_{i:04d}',
                    'name': f'Patient {i}',
                    'birth_date': f'19{self.random_state.randint(50,99)}-{self.random_state.randint(1,12):02d}-{self.random_state.randint(1,28):02d}'
                }

                # High heterogeneity in medical_history
                if self.random_state.random() > 0.4:
                    doc['medical_history'] = [
                        {'condition': 'hypertension', 'year': 2020},
                        {'condition': 'diabetes', 'year': 2022}
                    ][:self.random_state.randint(1, 3)]

                # Union type example
                if self.random_state.random() > 0.5:
                    contact_type = self.random_state.choice(['object', 'string'])
                    if contact_type == 'object':
                        doc['emergency_contact'] = {
                            'name': 'John Doe',
                            'phone': '555-0100'
                        }
                    else:
                        doc['emergency_contact'] = 'John Doe: 555-0100'

            elif doc_type == 'treatment':
                doc = {
                    'treatment_id': f'treat_{i:05d}',
                    'patient_id': f'pat_{self.random_state.randint(0, 150):04d}',
                    'medications': [
                        {'name': 'MedA', 'dose': '50mg'},
                        {'name': 'MedB', 'dose': '100mg'}
                    ]
                }

                # Complex union type with multiple variants
                dosage_type = self.random_state.choice(['float', 'string', 'object'], p=[0.5, 0.3, 0.2])
                if dosage_type == 'float':
                    doc['dosage'] = self.random_state.uniform(1, 500)
                elif dosage_type == 'string':
                    doc['dosage'] = f"{self.random_state.randint(1, 500)}mg"
                else:
                    doc['dosage'] = {
                        'amount': self.random_state.randint(1, 500),
                        'unit': 'mg',
                        'frequency': 'daily'
                    }

            documents.append(doc)

        return documents
